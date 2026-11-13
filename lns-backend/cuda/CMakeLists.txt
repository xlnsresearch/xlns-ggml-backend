# LNSCUDA backend: LNS arithmetic on CUDA GPUs through xlnscuda.
#
# Needs only the CUDA toolkit (nvcc + cudart), not ggml's own CUDA backend.
# Useful cache variables:
#   GGML_LNS16_SUM             base|monte|lpvip|sum32 (shared with LNS16)
#   XLNSCUDA_DIR               xlnscuda checkout (default: next to xlnscpp)
#   CMAKE_CUDA_ARCHITECTURES   default "native" (the GPUs in this machine)
#   CMAKE_CUDA_HOST_COMPILER   a host compiler your nvcc accepts, e.g. g++-12
#                              for CUDA 12.0 (or set CUDAHOSTCXX)

if (GGML_BACKEND_DL)
    # GGML_BACKEND_DL_IMPL exports one registry per library; this one has two.
    message(FATAL_ERROR "GGML_LNSCUDA does not support GGML_BACKEND_DL=ON")
endif()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86)
    endif()
endif()

include(CheckLanguage)
check_language(CUDA)
if (NOT CMAKE_CUDA_COMPILER)
    # check_language caches the failure; drop it so a fixed environment
    # (e.g. a supported host compiler) is picked up on the next configure.
    unset(CMAKE_CUDA_COMPILER CACHE)
    message(FATAL_ERROR "GGML_LNSCUDA=ON but nvcc is missing or rejects the host compiler "
                        "'${CMAKE_CUDA_HOST_COMPILER}'. Install the CUDA toolkit, or pass a host "
                        "compiler it supports: -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 (or set CUDAHOSTCXX)")
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

ggml_add_backend_library(ggml-lnscuda
    ggml-lns-cuda.cpp
    lns-cuda-context.cpp
    lns-cuda-ops.cpp
    lns-cuda-xlns16.cu
    lns-cuda-xlns32.cu
)

include(${CMAKE_CURRENT_SOURCE_DIR}/../ggml-lns-common/lns-common.cmake)
lns_backend_common_setup(ggml-lnscuda)

if (NOT DEFINED XLNSCUDA_DIR)
    set(XLNSCUDA_DIR "${XLNSCPP_DIR}/../xlnscuda")
endif()
if (NOT EXISTS "${XLNSCUDA_DIR}/xlns16d.cu")
    message(FATAL_ERROR "GGML_LNSCUDA: XLNSCUDA_DIR='${XLNSCUDA_DIR}' is not an xlnscuda checkout; "
                        "pass -DXLNSCUDA_DIR=/path/to/xlnscuda (run 'git submodule update --init' first)")
endif()
# The backend relies on xlnscuda's device lookup tables and its Monte / LPVIP
# reductions (xlns16d_monte.cu, xlns32d_lpvip.cu). Fail at configure time, not
# with compile errors.
file(READ "${XLNSCUDA_DIR}/xlns16d.cu" LNS_XLNS16D_SOURCE)
if (NOT LNS_XLNS16D_SOURCE MATCHES "xlns16d_table_init" OR NOT EXISTS "${XLNSCUDA_DIR}/xlns16d_monte.cu"
    OR NOT EXISTS "${XLNSCUDA_DIR}/xlns32d_lpvip.cu")
    message(FATAL_ERROR "GGML_LNSCUDA: ${XLNSCUDA_DIR} lacks the device tables or the Monte/LPVIP reductions; "
                        "update the xlnscuda submodule")
endif()
target_include_directories(ggml-lnscuda PRIVATE "${XLNSCUDA_DIR}")

# xlns16 summation mode (GGML_LNS16_SUM, shared with LNS16): only the xlns16
# translation unit gets its definitions.
lns16_sum_mode_defines(LNS_CUDA16_MODE_DEFS)
if (LNS_CUDA16_MODE_DEFS)
    set_property(SOURCE lns-cuda-xlns16.cu APPEND PROPERTY COMPILE_DEFINITIONS ${LNS_CUDA16_MODE_DEFS})
endif()

# LNS16 owns the Monte random state that LNSCUDA16 draws from (ggml-lns.h).
if (GGML_LNS)
    target_link_libraries(ggml-lnscuda PRIVATE ggml-lns)
    target_compile_definitions(ggml-lnscuda PRIVATE GGML_LNSCUDA_HAS_LNS16)
elseif (GGML_LNS16_SUM STREQUAL "monte")
    message(FATAL_ERROR "GGML_LNSCUDA with GGML_LNS16_SUM=monte needs GGML_LNS=ON")
endif()

# Static cudart: binaries built with LNSCUDA still start on machines without
# the CUDA toolkit (the LNSCUDA devices then report no GPU and stay hidden).
target_link_libraries(ggml-lnscuda PRIVATE CUDA::cudart_static)

# The xlnscpp/xlnscuda sources are written as C and trip many warnings that
# are not ours to fix; keep nvcc quiet about them.
target_compile_options(ggml-lnscuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=declared_but_not_referenced>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=set_but_not_used>)
