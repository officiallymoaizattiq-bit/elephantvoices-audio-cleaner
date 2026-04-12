"""
2D U-Net for elephant-rumble segmentation on spectrograms.

Two operating modes, controlled by the `embedding_dim` constructor argument:

* `embedding_dim = 0` (default, backward-compatible):
      Semantic segmentation. forward(x) -> [B, 1, F, T] raw logits. Train with
      BCEWithLogitsLoss. This is the original rumble-vs-noise model.

* `embedding_dim > 0`:
      Instance segmentation via Deep Clustering (Hershey et al. 2016).
      forward(x) -> (mask_logits, embeddings) where embeddings has shape
      [B, D, F, T] and is L2-normalised along the channel dim. At inference,
      k-means on the active bins' embeddings splits overlapping callers into
      distinct instance masks. Train with losses.InstanceSegLoss.

Architectural choices (the non-obvious stuff):

* Depth 4. With real data shape (2048, 32) the bottleneck becomes (128, 2);
  going deeper collapses the time axis to 1.
* Encoder features: 64 -> 128 -> 256 -> 512, bottleneck 1024. ~31 M params.
* Every conv is followed by BatchNorm2d + ReLU to keep training stable.
* Upsampling uses ConvTranspose2d stride 2. The Up module pads if decoder
  output is slightly smaller than its skip tensor.
* Instance head is a single 1x1 conv that projects the decoder features down
  to D channels, followed by a per-bin L2 normalisation. Keeping it shallow
  is deliberate: the U-Net decoder already produces rich features, and a
  deeper embedding head would just bloat params.
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2. The base block used throughout the U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool-then-DoubleConv encoder step."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """ConvTranspose2d-then-DoubleConv decoder step with skip concat."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Halve channel count during upsampling; the skip tensor brings the
        # other half so the DoubleConv still sees `in_channels` total.
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad in case of off-by-one from odd spatial dims on the encoder side.
        # Pad order is (left, right, top, bottom).
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        if diff_h or diff_w:
            x = F.pad(
                x,
                [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
            )
        return self.conv(torch.cat([skip, x], dim=1))


UNetOutput = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class UNet(nn.Module):
    """Depth-4 2D U-Net with an optional instance-segmentation embedding head.

    Args:
        in_channels:   input channels (1 for spectrograms).
        out_channels:  mask-head output channels (1 for rumble-vs-noise).
        base_features: feature ladder multiplier (64 gives the classic U-Net).
        embedding_dim: if > 0, attach a Deep Clustering embedding head that
                       emits D-dimensional L2-normalised per-bin embeddings.
                       Set to 0 for the original single-mask behaviour.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 64,
        embedding_dim: int = 0,
    ) -> None:
        super().__init__()
        f = base_features  # 64, 128, 256, 512, 1024
        self.embedding_dim = embedding_dim

        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16)

        self.up1 = Up(f * 16, f * 8)
        self.up2 = Up(f * 8, f * 4)
        self.up3 = Up(f * 4, f * 2)
        self.up4 = Up(f * 2, f)

        # Rumble/noise mask head (unchanged from the original model).
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

        # Instance embedding head. Only instantiated when requested so that
        # loading the original (mask-only) state_dict still matches exactly.
        if embedding_dim > 0:
            self.out_embed = nn.Conv2d(f, embedding_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> UNetOutput:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        mask_logits = self.out_conv(x)

        if self.embedding_dim > 0:
            # Per-bin embedding, L2 normalised along the channel dim. The
            # normalisation is what turns dot products into cosine similarity
            # for the Deep Clustering loss and k-means at inference.
            embeddings = self.out_embed(x)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return mask_logits, embeddings

        return mask_logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> UNetOutput:
        """Return sigmoid(mask). In instance mode also returns the embeddings."""
        self.eval()
        out = self.forward(x)
        if isinstance(out, tuple):
            mask_logits, embeddings = out
            return torch.sigmoid(mask_logits), embeddings
        return torch.sigmoid(out)


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    batch, channels, height, width = 2, 1, 256, 256
    dummy = torch.randn(batch, channels, height, width)

    # ---- 1. Semantic-segmentation mode (unchanged original behaviour) ------
    model = UNet(in_channels=channels, out_channels=1, embedding_dim=0).eval()
    with torch.no_grad():
        logits = model(dummy)
    assert logits.shape == dummy.shape
    print(f"[unet] semantic mode  | logits {tuple(logits.shape)} | "
          f"params {sum(p.numel() for p in model.parameters()):,}")

    # ---- 2. Instance-segmentation mode with a 20-dim embedding head --------
    inst_model = UNet(
        in_channels=channels, out_channels=1, embedding_dim=20
    ).eval()
    with torch.no_grad():
        mask_logits, embeddings = inst_model(dummy)
    assert mask_logits.shape == dummy.shape
    assert embeddings.shape == (batch, 20, height, width)
    norms = embeddings.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        "Embeddings must be L2-normalised along the channel axis"
    )
    print(f"[unet] instance mode  | mask {tuple(mask_logits.shape)} | "
          f"emb {tuple(embeddings.shape)} | "
          f"params {sum(p.numel() for p in inst_model.parameters()):,}")
    print("[unet] OK - both modes produce the expected shapes.")
