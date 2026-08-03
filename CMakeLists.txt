cmake_minimum_required(VERSION 3.20)
project(cim-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

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

  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${MLIR_INCLUDE_DIRS})
  link_directories(${LLVM_BUILD_LIBRARY_DIR})
  add_definitions(${LLVM_DEFINITIONS})
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

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
