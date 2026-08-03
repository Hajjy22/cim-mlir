# cim-bench needs only the core, so it builds without MLIR.
add_subdirectory(cim-bench)

if(CIM_ENABLE_MLIR)
  add_subdirectory(cim-opt)
endif()
