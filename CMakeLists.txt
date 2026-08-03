cmake_minimum_required(VERSION 3.20)
project(cim-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# clang-tidy and cppcheck both read compile_commands.json.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CIMBuildFlags)

# The project splits in two:
#
#   * A core (lib/Placement, lib/Target, runtime/, tools/cim-bench) with no
#     LLVM or MLIR dependency at all. This is where the placement algorithm
#     -- the actual contribution, see docs/abstraction.md -- lives, and it
#     builds and tests anywhere a C++17 compiler exists.
#
#   * The MLIR layer (include/cim/Dialect, lib/Dialect, lib/Transforms,
#     tools/cim-opt, the FileCheck suite), which needs a real LLVM/MLIR
#     install.
#
# MLIR is therefore optional: without it you still get the core library,
# its unit tests, and a working cim-bench, rather than a configure error.
option(CIM_ENABLE_MLIR "Build the MLIR dialect, passes, and cim-opt" ON)
option(CIM_INCLUDE_TESTS "Build the cim-mlir test suites" ON)

# Put every tool in <build>/bin. Without this, add_llvm_executable and
# add_executable scatter binaries across their source-relative build
# directories, and the lit suite cannot find cim-opt.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)

if(CIM_ENABLE_MLIR)
  find_package(MLIR CONFIG)
  if(NOT MLIR_FOUND)
    message(WARNING
      "MLIR not found: building the core only (placement engine, target "
      "reader, runtime, cim-bench). Pass -DMLIR_DIR=<prefix>/lib/cmake/mlir "
      "and -DLLVM_DIR=<prefix>/lib/cmake/llvm to build the dialect, the "
      "lowering passes, and cim-opt.")
    set(CIM_ENABLE_MLIR OFF)
  endif()
endif()

if(CIM_ENABLE_MLIR)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  # SYSTEM: upstream MLIR/LLVM headers emit warnings we neither own nor can
  # fix, and -Werror would turn every one of them into our build failure.
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
  include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
  link_directories(${LLVM_BUILD_LIBRARY_DIR})
  add_definitions(${LLVM_DEFINITIONS})
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# Also SYSTEM: this is where TableGen writes its generated .inc files, which
# are not ours to keep warning-clean either.
include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/include)

if(CIM_ENABLE_MLIR)
  add_subdirectory(include)
endif()

add_subdirectory(lib)
add_subdirectory(runtime)
add_subdirectory(tools)

if(CIM_INCLUDE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
