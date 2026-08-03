add_llvm_executable(cim-opt cim-opt.cpp)

llvm_update_compile_flags(cim-opt)

# Only the dialects cim-opt actually registers -- see the note at the top of
# cim-opt.cpp for why this is not registerAllDialects().
target_link_libraries(cim-opt PRIVATE
  MLIRCIM
  MLIRCIMTransforms
  MLIROptLib
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRSCFDialect
)

mlir_check_all_link_libraries(cim-opt)
