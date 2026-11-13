ggml_add_backend_library(ggml-lns
    ggml-lns.cpp
    lns-ops.cpp
)

include(${CMAKE_CURRENT_SOURCE_DIR}/../ggml-lns-common/lns-common.cmake)
lns_backend_common_setup(ggml-lns)

# xlnscpp uses a header-inclusion model: #include "xlns16.cpp" with defines.
# xlns16_table enables the 65536-entry O(1) float<->xlns16 lookup table.
# xlns16_alt enables the reduced-branching sb/db addition path.
# xlns16_ideal is intentionally NOT set (table approximation, consistent with a realistic hardware target).
target_compile_definitions(ggml-lns PRIVATE xlns16_table xlns16_alt)

# Summation variant (GGML_LNS16_SUM, GGML_LNS16_MONTE_BYLPVIP; lns-common.cmake):
#   base — xlns16_add, monte — Monte-Carlo dither, lpvip — 32-bit LPVIP
#   accumulator, sum32 — xlns32 accumulator
lns16_sum_mode_defines(LNS16_MODE_DEFS)
if (LNS16_MODE_DEFS)
    target_compile_definitions(ggml-lns PRIVATE ${LNS16_MODE_DEFS})
endif()
