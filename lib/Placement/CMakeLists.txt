# The placement/eviction engine and its cost model. Plain C++17 with no
# LLVM or MLIR dependency so it can be built and tested standalone -- see
# the rationale in the top-level CMakeLists.txt.
add_library(cimPlacement STATIC
  Placement.cpp
  CostReport.cpp
  Workloads.cpp
  WorkloadJSON.cpp
)

target_include_directories(cimPlacement PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(cimPlacement PUBLIC cimTarget cim_build_flags)

set_target_properties(cimPlacement PROPERTIES POSITION_INDEPENDENT_CODE ON)
