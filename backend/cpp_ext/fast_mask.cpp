// fast_mask.cpp
// -----------------------------------------------------------------------------
// OpenMP-parallel, GIL-free element-wise multiply between a complex STFT and a
// real-valued mask. Built as a pybind11 extension called `fast_mask_ext`.
//
// Why this exists: after the U-Net produces a soft mask, inference.py applies
// it to the complex STFT with `mask * complex_stft`. NumPy already releases
// the GIL during its ufunc loops, but doing it in C++ lets us:
//   * release the GIL explicitly for the full duration of the op,
//   * parallelise with OpenMP across physical cores,
//   * compile with -O3 -ffast-math -funroll-loops for tight SIMD codegen.
//
// The API is intentionally minimal:
//   fast_mask_ext.apply_mask(complex_stft, mask) -> complex_stft * mask
// -----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <complex>
#include <cstddef>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

using complex64 = std::complex<float>;
using CArray = py::array_t<complex64, py::array::c_style | py::array::forcecast>;
using FArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

static CArray apply_mask(CArray stft, FArray mask) {
    // ---- 1. Validate shapes -------------------------------------------------
    auto stft_info = stft.request();
    auto mask_info = mask.request();

    if (stft_info.ndim != 2 || mask_info.ndim != 2) {
        throw std::invalid_argument("fast_mask.apply_mask: both inputs must be 2D arrays");
    }
    if (stft_info.shape[0] != mask_info.shape[0] ||
        stft_info.shape[1] != mask_info.shape[1]) {
        throw std::invalid_argument(
            "fast_mask.apply_mask: stft and mask must have identical shape");
    }

    const std::ptrdiff_t rows = static_cast<std::ptrdiff_t>(stft_info.shape[0]);
    const std::ptrdiff_t cols = static_cast<std::ptrdiff_t>(stft_info.shape[1]);
    const std::ptrdiff_t total = rows * cols;

    // ---- 2. Allocate output (same shape / dtype as input STFT) --------------
    CArray out({rows, cols});
    auto out_info = out.request();

    const complex64* __restrict__ stft_ptr =
        static_cast<const complex64*>(stft_info.ptr);
    const float* __restrict__ mask_ptr =
        static_cast<const float*>(mask_info.ptr);
    complex64* __restrict__ out_ptr =
        static_cast<complex64*>(out_info.ptr);

    // ---- 3. GIL-free parallel loop ------------------------------------------
    // Everything past this point touches only raw pointers, so it's safe to
    // release the GIL and let other Python threads run.
    {
        py::gil_scoped_release release;

        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < total; ++i) {
            const float m = mask_ptr[i];
            const complex64 z = stft_ptr[i];
            // Real scalar * complex = scale magnitude, preserve phase.
            out_ptr[i] = complex64(z.real() * m, z.imag() * m);
        }
    }

    return out;
}

PYBIND11_MODULE(fast_mask_ext, m) {
    m.doc() = "OpenMP-accelerated complex STFT x real mask element-wise multiply.";

    m.def("apply_mask", &apply_mask,
          py::arg("stft"), py::arg("mask"),
          R"pbdoc(
              Element-wise multiply a complex STFT by a real mask.

              Args:
                  stft: complex64 ndarray of shape (F, T)
                  mask: float32 ndarray of shape (F, T)

              Returns:
                  New complex64 ndarray of shape (F, T) containing stft * mask.

              The multiply is performed in C++ with the GIL released; if the
              module was built against OpenMP it will also fan out across
              every physical core.
          )pbdoc");

#ifdef _OPENMP
    m.attr("openmp_enabled") = true;
    m.attr("max_threads") = omp_get_max_threads();
#else
    m.attr("openmp_enabled") = false;
    m.attr("max_threads") = 1;
#endif
}
