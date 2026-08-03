# Build-instrumentation flags, carried on one INTERFACE target rather than
# mutated into CMAKE_CXX_FLAGS globally. Every cim target links this PUBLIC so
# that sanitizer *link* options propagate to executables -- linking a
# sanitized library into an uninstrumented binary fails at runtime in
# confusing ways.
#
# All options default OFF: a plain release build is byte-for-byte unaffected.

option(CIM_ENABLE_COVERAGE "Instrument for gcov coverage" OFF)
option(CIM_ENABLE_WERROR   "Treat warnings as errors" OFF)
set(CIM_SANITIZER "" CACHE STRING
    "Sanitizers to enable, e.g. address,undefined (empty = none)")

add_library(cim_build_flags INTERFACE)

if(CIM_ENABLE_WERROR)
  target_compile_options(cim_build_flags INTERFACE
    -Wall -Wextra -Wshadow -Wno-unused-parameter -Werror)
endif()

if(CIM_SANITIZER)
  set(_cim_san_flags -fsanitize=${CIM_SANITIZER})

  # MLIR/LLVM are built -fno-rtti, so UBSan's vptr check has no type
  # information to work with and reports false positives on every dyn_cast.
  if(CIM_SANITIZER MATCHES "undefined")
    list(APPEND _cim_san_flags -fno-sanitize=vptr)
  endif()

  target_compile_options(cim_build_flags INTERFACE
    ${_cim_san_flags}
    -fno-omit-frame-pointer
    -fno-sanitize-recover=all   # a sanitizer report must fail the test, not warn
    -g)
  target_link_options(cim_build_flags INTERFACE ${_cim_san_flags})

  message(STATUS "cim-mlir: sanitizers enabled: ${CIM_SANITIZER}")
endif()

if(CIM_ENABLE_COVERAGE)
  # gcov/gcc only. Clang source-based coverage (-fprofile-instr-generate)
  # needs compiler-rt, which is not guaranteed present alongside a distro
  # MLIR install.
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(WARNING
      "CIM_ENABLE_COVERAGE expects GCC (gcov). Current compiler is "
      "${CMAKE_CXX_COMPILER_ID}; coverage output may be unusable.")
  endif()
  target_compile_options(cim_build_flags INTERFACE --coverage -O0 -g)
  target_link_options(cim_build_flags INTERFACE --coverage)
  message(STATUS "cim-mlir: gcov coverage instrumentation enabled")
endif()
