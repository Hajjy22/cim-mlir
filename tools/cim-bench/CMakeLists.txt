add_executable(cim-bench cim-bench.cpp)

target_link_libraries(cim-bench PRIVATE
  cimPlacement
  cimTarget
  cim_build_flags
)
