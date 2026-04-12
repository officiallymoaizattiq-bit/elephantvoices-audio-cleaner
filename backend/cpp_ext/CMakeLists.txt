cmake_minimum_required(VERSION 3.18)
project(fast_mask_ext LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Build configuration
# -----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -----------------------------------------------------------------------------
# pybind11 (installed via `pip install pybind11`)
# -----------------------------------------------------------------------------

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# -----------------------------------------------------------------------------
# OpenMP
# -----------------------------------------------------------------------------
# Tricky on Apple Silicon. Three gotchas:
#   1. Apple Clang doesn't ship an OpenMP runtime, so we need Homebrew's libomp
#      or similar for the omp.h headers.
#   2. PyTorch on macOS statically embeds its own libomp at install_name
#      /opt/llvm-openmp/lib/libomp.dylib. If we link against Homebrew's libomp
#      (install_name /opt/homebrew/opt/libomp/lib/libomp.dylib), dyld sees two
#      separate install names and loads TWO copies of the runtime into the
#      same process. Both libraries claim symbols like omp_get_max_threads,
#      and the result is a runtime segfault the first time both runtimes try
#      to touch shared state.
#   3. Fix: detect torch's bundled libomp at configure time and link the
#      extension against that exact file. fast_mask_ext then records torch's
#      install name as its dependency, so when inference.py imports torch
#      first (which it does), dyld has already resolved that install name
#      and our extension reuses the same runtime instance.
# -----------------------------------------------------------------------------

if(APPLE)
    # Probe torch's libomp location via the configured Python interpreter.
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c
                "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libomp.dylib'))"
        OUTPUT_VARIABLE TORCH_LIBOMP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TORCH_LIBOMP_RC
        ERROR_QUIET
    )

    # Probe Homebrew libomp for the headers.
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE HOMEBREW_LIBOMP_RC
        ERROR_QUIET
    )

    if(HOMEBREW_LIBOMP_RC EQUAL 0 AND HOMEBREW_LIBOMP_PREFIX)
        set(OMP_INCLUDE_DIR "${HOMEBREW_LIBOMP_PREFIX}/include")
    endif()

    if(TORCH_LIBOMP_RC EQUAL 0 AND EXISTS "${TORCH_LIBOMP}" AND OMP_INCLUDE_DIR)
        # Shared runtime with PyTorch - the safe path.
        message(STATUS "Linking OpenMP against torch's libomp: ${TORCH_LIBOMP}")
        set(OpenMP_C_FLAGS "-Xclang -fopenmp -I${OMP_INCLUDE_DIR}")
        set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I${OMP_INCLUDE_DIR}")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${TORCH_LIBOMP}")
    elseif(HOMEBREW_LIBOMP_PREFIX AND EXISTS "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
        # Homebrew-only path. Works if torch isn't in the process - for a
        # standalone extension import this is fine, but if you plan to call
        # this module from the same Python process that imports torch, you
        # almost certainly want the torch-libomp path above.
        message(WARNING
            "Linking OpenMP against Homebrew libomp; will conflict with torch "
            "if both are loaded in the same Python process.")
        set(OpenMP_C_FLAGS "-Xclang -fopenmp -I${OMP_INCLUDE_DIR}")
        set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I${OMP_INCLUDE_DIR}")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

find_package(OpenMP)

# -----------------------------------------------------------------------------
# Target
# -----------------------------------------------------------------------------

pybind11_add_module(fast_mask_ext fast_mask.cpp)

target_compile_options(fast_mask_ext PRIVATE
    -O3
    -ffast-math
    -funroll-loops
    -Wall
    -Wextra
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(fast_mask_ext PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP: ENABLED - fast_mask_ext will run in parallel")
else()
    message(WARNING "OpenMP not found - fast_mask_ext will build serial-only")
endif()

# -----------------------------------------------------------------------------
# rpath + install_name rewrite
# -----------------------------------------------------------------------------
# torch's libomp.dylib ships with install_name /opt/llvm-openmp/lib/libomp.dylib,
# a path that doesn't exist on disk - it only resolves when torch has already
# loaded that file into the process. Left alone, importing fast_mask_ext
# without torch preloaded blows up with "Library not loaded".
#
# Fix: embed torch/lib as an rpath, then use install_name_tool to rewrite
# fast_mask_ext's own load command so it looks for the library via @rpath.
# Result: dyld finds torch's libomp.dylib via rpath, loads it, and (if torch
# is also loaded) reuses the existing runtime instance because both modules
# end up keyed on the same install_name.
# -----------------------------------------------------------------------------

if(APPLE AND TORCH_LIBOMP)
    get_filename_component(TORCH_LIB_DIR "${TORCH_LIBOMP}" DIRECTORY)
    set_target_properties(fast_mask_ext PROPERTIES
        BUILD_RPATH "${TORCH_LIB_DIR}"
        INSTALL_RPATH "${TORCH_LIB_DIR}"
        MACOSX_RPATH ON
    )

    add_custom_command(TARGET fast_mask_ext POST_BUILD
        COMMAND install_name_tool -change
                /opt/llvm-openmp/lib/libomp.dylib
                @rpath/libomp.dylib
                $<TARGET_FILE:fast_mask_ext>
        COMMENT "Rewriting libomp load command to @rpath/libomp.dylib"
    )
endif()

# Drop the compiled .so next to the source so Python can import it without
# any sys.path manipulation in user code (inference.py adds this directory
# to sys.path at load time).
add_custom_command(TARGET fast_mask_ext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:fast_mask_ext>
            ${CMAKE_CURRENT_SOURCE_DIR}/$<TARGET_FILE_NAME:fast_mask_ext>
    COMMENT "Copying fast_mask_ext module into source directory"
)
