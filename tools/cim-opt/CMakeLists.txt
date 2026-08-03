add_llvm_executable(cim-opt cim-opt.cpp)

llvm_update_compile_flags(cim-opt)

target_link_libraries(cim-opt PRIVATE
  MLIRCIM
  MLIRCIMTransforms
  MLIROptLib
  MLIRIR
)

mlir_check_all_link_libraries(cim-opt)
