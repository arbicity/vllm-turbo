# Applies PATCH_DIR/*.patch to SOURCE_DIR, skipping any patch already applied.
# Invoked as a FetchContent PATCH_COMMAND, which re-runs on every re-configure.
if(NOT SOURCE_DIR OR SOURCE_DIR MATCHES "^<")
  set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

file(GLOB _patches "${PATCH_DIR}/*.patch")
list(SORT _patches)
foreach(_patch IN LISTS _patches)
  execute_process(
    COMMAND git apply --reverse --check "${_patch}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE _already
    OUTPUT_QUIET ERROR_QUIET)
  if(_already EQUAL 0)
    message(STATUS "vllm-flash-attn patch already applied: ${_patch}")
    continue()
  endif()
  execute_process(
    COMMAND git apply "${_patch}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE _rc
    ERROR_VARIABLE _err)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "Failed to apply ${_patch}: ${_err}")
  endif()
  message(STATUS "vllm-flash-attn patch applied: ${_patch}")
endforeach()
