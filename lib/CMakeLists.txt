# Core: no LLVM/MLIR dependency, always built.
add_subdirectory(Placement)
add_subdirectory(Target)

# MLIR layer.
if(CIM_ENABLE_MLIR)
  add_subdirectory(Dialect)
  add_subdirectory(Transforms)
  add_subdirectory(Interpreter)
endif()
