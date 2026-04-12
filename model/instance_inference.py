"""
Instance-segmentation inference via HDBSCAN on double-thresholded embeddings.

Pipeline:
    spec -> UNet(instance mode) -> (mask_logits, embeddings)
        -> double-threshold: sigmoid(mask) > T1 AND mag_norm > (max - headroom)
        -> gather 20D embeddings at surviving bins only
        -> subsample to 15k points if too many (avoids O(n^2) in HDBSCAN)
        -> HDBSCAN(min_cluster_size=50) -> cluster labels + noise label -1
        -> KNeighborsClassifier propagates labels to the full active set
        -> bins with label -1 are EXCLUDED from all output masks (noise)
        -> K = number of unique valid clusters found by HDBSCAN

Why HDBSCAN over KMeans:
    * KMeans forces exactly K clusters and has no concept of noise. With a
      saturated mask head, KMeans splits background-noise bins into fake
      "callers." HDBSCAN's noise label (-1) naturally filters those out.
    * HDBSCAN infers K from the density structure of the embedding space.
      Dense clusters of embeddings become callers; sparse regions become noise.
    * No silhouette-based post-hoc K estimation needed.

Why double-thresholding instead of the mask alone:
    The baseline mask head is saturated (~99% of bins active at sigmoid > 0.5).
    Adding a magnitude-based gate (only bins within 15 dB of the peak) ensures
    we only cluster the high-energy elephant harmonics, not the background drone
    from generators/airplanes that the mask can't distinguish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from sklearn.cluster import HDBSCAN
from sklearn.neighbors import KNeighborsClassifier


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP_DB: float = 80.0               # Must match dataset.py normalisation
DB_HEADROOM: float = 10.0          # Only bins within 10 dB of peak survive (strict)
MASK_THRESHOLD: float = 0.5        # Sigmoid gate
MIN_ACTIVE_BINS: int = 64          # Below this -> degenerate, return K=1
MAX_HDBSCAN_POINTS: int = 5000     # Subsample cap — smaller = faster clustering
HDBSCAN_MIN_CLUSTER: int = 500     # Minimum points in a cluster within the subsample
HDBSCAN_MIN_SAMPLES: int = 50      # Core-point neighbourhood density
KNN_NEIGHBORS: int = 5             # For propagating labels to full active set
MIN_CLUSTER_BINS: int = 500        # Drop clusters smaller than this in final output
# Post-HDBSCAN merge: if two cluster centroids are closer than this euclidean
# distance (on L2-normalised embeddings), collapse them into one caller.
# At 0.5 euclidean on unit vectors, cosine_similarity >= 0.875. The embedding
# head was trained exclusively on K=2 mixtures so it always produces at least
# 2 dense sub-regions even for a single caller. This merge catches that bias.
CLUSTER_MERGE_DISTANCE: float = 0.65


@dataclass
class InstanceSegResult:
    """Container returned by `separate_sources`."""
    source_masks: np.ndarray             # [K, F, T] binary per-source masks (noise excluded)
    rumble_mask: np.ndarray              # [F, T] union mask (double-thresholded active bins)
    embeddings: np.ndarray               # [D, F, T] raw embeddings from the model
    cluster_labels: Optional[np.ndarray] # [num_active] HDBSCAN/KNN labels (-1 = noise)
    detected_k: int                      # Number of valid clusters (callers)
    num_active_bins: int                 # Bins that passed the double threshold
    num_noise_bins: int                  # Bins labeled -1 by HDBSCAN (excluded)
    hdbscan_info: Dict = field(default_factory=dict)


def _double_threshold(
    mask_prob: np.ndarray,
    spec_norm: np.ndarray,
    mask_threshold: float,
    db_headroom: float,
) -> np.ndarray:
    """Return a boolean (F, T) mask where BOTH conditions hold:
        1. sigmoid(mask_logits) > mask_threshold   (model confidence)
        2. spec_norm > spec_norm.max() - headroom  (magnitude energy)

    The second condition cuts through the saturated mask by requiring bins
    to be within `db_headroom` dB of the loudest bin in the spectrogram.
    In the [0, 1] normalised space (where 0 = -80 dB, 1 = 0 dB), the
    threshold becomes `max - headroom / TOP_DB`.
    """
    if float(spec_norm.max()) < 1e-6:
        return np.zeros_like(mask_prob, dtype=bool)
    mask_active = mask_prob > mask_threshold
    db_thresh = float(spec_norm.max()) - db_headroom / TOP_DB
    mag_active = spec_norm > db_thresh
    return mask_active & mag_active


def _merge_close_clusters(
    labels: np.ndarray,
    active_vectors: np.ndarray,
    merge_distance: float,
) -> np.ndarray:
    """Merge HDBSCAN clusters whose centroids are within `merge_distance`.

    The embedding head was trained only on K=2 synthetic mixtures, so it
    always produces at least 2 dense sub-regions in the 20D embedding space
    even for a single caller. This post-hoc merge collapses sub-clusters
    that are genuinely close in embedding space back into one caller.

    Uses a simple union-find over pairwise centroid distances. O(K^2) where K
    is the number of raw clusters — typically 1-6, so it's instant.
    """
    valid = sorted(set(labels) - {-1})
    if len(valid) <= 1:
        return labels

    centroids = np.stack([active_vectors[labels == k].mean(axis=0) for k in valid])

    # Union-find
    parent = {k: k for k in valid}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            if dist < merge_distance:
                union(valid[i], valid[j])

    # Relabel so merged clusters share the same label, starting from 0
    new_labels = labels.copy()
    root_to_new: Dict[int, int] = {}
    next_id = 0
    for k in valid:
        root = find(k)
        if root not in root_to_new:
            root_to_new[root] = next_id
            next_id += 1
        new_labels[labels == k] = root_to_new[root]

    # Keep noise as -1
    new_labels[labels == -1] = -1
    return new_labels


def _run_hdbscan_with_subsampling(
    active_vectors: np.ndarray,
    active_coords: np.ndarray,
    max_points: int,
    min_cluster_size: int,
    min_samples: int,
    knn_neighbors: int,
    random_state: int,
) -> np.ndarray:
    """Cluster embeddings via HDBSCAN, subsampling if the active set is large.

    Strategy:
        1. If n_active <= max_points, fit HDBSCAN directly on all points.
        2. Otherwise, randomly sample `max_points` active embeddings, fit
           HDBSCAN on that sample to discover the cluster structure, then
           use a KNeighborsClassifier (trained on the non-noise sample
           points) to propagate labels to every active bin.

    Returns a label array of shape (n_active,) where -1 = noise.
    """
    rng = np.random.default_rng(random_state)
    n_active = active_vectors.shape[0]

    if n_active <= max_points:
        hdb = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
        )
        return hdb.fit_predict(active_vectors)

    # --- Subsampled path ---
    sample_idx = rng.choice(n_active, max_points, replace=False)
    sample_vectors = active_vectors[sample_idx]

    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    sample_labels = hdb.fit_predict(sample_vectors)

    n_valid = int((sample_labels >= 0).sum())
    if n_valid < knn_neighbors:
        return np.full(n_active, -1, dtype=np.int64)

    # Train KNN on ALL sample points INCLUDING noise (-1) so the classifier
    # CAN predict -1 for bins that sit near noise regions. Without this,
    # every active bin gets force-assigned to a cluster and the noise
    # filtering documented in the module docstring never actually fires.
    knn = KNeighborsClassifier(
        n_neighbors=min(knn_neighbors, len(sample_vectors)),
        metric="cosine",
    )
    knn.fit(sample_vectors, sample_labels)
    all_labels = knn.predict(active_vectors)

    return all_labels.astype(np.int64)


@torch.no_grad()
def separate_sources(
    model,
    spec: torch.Tensor,
    mask_threshold: float = MASK_THRESHOLD,
    db_headroom: float = DB_HEADROOM,
    min_active_bins: int = MIN_ACTIVE_BINS,
    max_hdbscan_points: int = MAX_HDBSCAN_POINTS,
    hdbscan_min_cluster: int = HDBSCAN_MIN_CLUSTER,
    hdbscan_min_samples: int = HDBSCAN_MIN_SAMPLES,
    knn_neighbors: int = KNN_NEIGHBORS,
    min_cluster_bins: int = MIN_CLUSTER_BINS,
    random_state: int = 0,
) -> InstanceSegResult:
    """Run a double-thresholded, HDBSCAN-clustered instance segmentation.

    Args:
        model:  UNet with embedding_dim > 0.
        spec:   [1, 1, F, T] normalised magnitude spectrogram.
        mask_threshold: sigmoid gate for the mask head.
        db_headroom: only keep bins within this many dB of the peak magnitude.
        min_active_bins: if fewer bins survive thresholding, return K=1.
        max_hdbscan_points: subsample to this many points before HDBSCAN.
        hdbscan_min_cluster: HDBSCAN min_cluster_size parameter.
        hdbscan_min_samples: HDBSCAN min_samples parameter.
        knn_neighbors: KNN neighbours for label propagation after subsampling.
        min_cluster_bins: drop clusters with fewer than this many bins.
        random_state: reproducibility seed.
    """
    if spec.dim() != 4 or spec.shape[0] != 1:
        raise ValueError(f"Expected spec shape [1, 1, F, T], got {tuple(spec.shape)}")

    model.eval()
    out = model(spec)
    if not isinstance(out, tuple):
        raise RuntimeError(
            "Model did not return embeddings. Use UNet(embedding_dim > 0)."
        )
    mask_logits, embeddings = out

    mask_prob = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()   # (F, T)
    spec_np = spec[0, 0].cpu().numpy()                           # (F, T)
    emb = embeddings[0].cpu().numpy()                             # (D, F, T)
    D, F_bins, T = emb.shape

    # --- Double threshold: mask confidence AND magnitude energy -------------
    active = _double_threshold(mask_prob, spec_np, mask_threshold, db_headroom)
    active_coords = np.argwhere(active)  # (N_active, 2)
    num_active = int(active_coords.shape[0])

    # --- Degenerate case: too few active bins -> single source ---------------
    if num_active < min_active_bins:
        single_mask = active.astype(np.float32)[np.newaxis, :, :]
        return InstanceSegResult(
            source_masks=single_mask,
            rumble_mask=active.astype(np.float32),
            embeddings=emb,
            cluster_labels=None,
            detected_k=1,
            num_active_bins=num_active,
            num_noise_bins=0,
            hdbscan_info={"reason": "too_few_active_bins"},
        )

    # --- Gather embeddings at active bins ------------------------------------
    active_vectors = emb[:, active_coords[:, 0], active_coords[:, 1]].T  # (N, D)

    # --- HDBSCAN clustering --------------------------------------------------
    labels = _run_hdbscan_with_subsampling(
        active_vectors=active_vectors,
        active_coords=active_coords,
        max_points=max_hdbscan_points,
        min_cluster_size=hdbscan_min_cluster,
        min_samples=hdbscan_min_samples,
        knn_neighbors=knn_neighbors,
        random_state=random_state,
    )

    # --- Merge clusters whose centroids are too close -------------------------
    labels = _merge_close_clusters(labels, active_vectors, CLUSTER_MERGE_DISTANCE)

    valid_labels = set(labels) - {-1}
    num_noise = int((labels == -1).sum())

    # If HDBSCAN found no clusters at all, everything is noise -> single source
    if not valid_labels:
        single_mask = active.astype(np.float32)[np.newaxis, :, :]
        return InstanceSegResult(
            source_masks=single_mask,
            rumble_mask=active.astype(np.float32),
            embeddings=emb,
            cluster_labels=labels,
            detected_k=1,
            num_active_bins=num_active,
            num_noise_bins=num_noise,
            hdbscan_info={"reason": "all_noise"},
        )

    # --- Build per-cluster masks, filtering out small clusters ---------------
    sorted_labels = sorted(valid_labels)
    kept_masks: list[np.ndarray] = []
    for lbl in sorted_labels:
        cluster_coords = active_coords[labels == lbl]
        if cluster_coords.shape[0] < min_cluster_bins:
            continue
        mask_k = np.zeros((F_bins, T), dtype=np.float32)
        mask_k[cluster_coords[:, 0], cluster_coords[:, 1]] = 1.0
        kept_masks.append(mask_k)

    # Fall back to single-source if all clusters were filtered out
    if not kept_masks:
        single_mask = active.astype(np.float32)[np.newaxis, :, :]
        return InstanceSegResult(
            source_masks=single_mask,
            rumble_mask=active.astype(np.float32),
            embeddings=emb,
            cluster_labels=labels,
            detected_k=1,
            num_active_bins=num_active,
            num_noise_bins=num_noise,
            hdbscan_info={"reason": "all_clusters_too_small"},
        )

    detected_k = len(kept_masks)
    source_masks = np.stack(kept_masks, axis=0)  # (K, F, T)

    return InstanceSegResult(
        source_masks=source_masks,
        rumble_mask=active.astype(np.float32),
        embeddings=emb,
        cluster_labels=labels,
        detected_k=detected_k,
        num_active_bins=num_active,
        num_noise_bins=num_noise,
        hdbscan_info={
            "n_clusters_raw": len(valid_labels),
            "n_clusters_kept": detected_k,
        },
    )


def apply_instance_masks_to_stft(
    source_masks: np.ndarray,
    complex_stft: np.ndarray,
) -> list[np.ndarray]:
    """Apply each per-source mask to the full-resolution complex STFT."""
    if source_masks.ndim != 3:
        raise ValueError(f"source_masks must be [K, F, T]; got {source_masks.shape}")
    K, F_crop, T = source_masks.shape
    F_full = complex_stft.shape[0]
    stfts: list[np.ndarray] = []
    for k in range(K):
        full_mask = np.zeros((F_full, T), dtype=np.float32)
        full_mask[:F_crop] = source_masks[k]
        stfts.append(full_mask.astype(np.complex64) * complex_stft)
    return stfts


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from unet import UNet

    torch.manual_seed(0)
    model = UNet(in_channels=1, out_channels=1, embedding_dim=20).eval()
    spec = torch.randn(1, 1, 2048, 32)
    result = separate_sources(model, spec)

    print(f"[hdbscan] detected_k     : {result.detected_k}")
    print(f"[hdbscan] num_active_bins: {result.num_active_bins}")
    print(f"[hdbscan] num_noise_bins : {result.num_noise_bins}")
    print(f"[hdbscan] source_masks   : {result.source_masks.shape}")
    print(f"[hdbscan] info           : {result.hdbscan_info}")
