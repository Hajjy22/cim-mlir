add_llvm_executable(cim-run cim-run.cpp)

llvm_update_compile_flags(cim-run)

target_link_libraries(cim-run PRIVATE
  MLIRCIM
  MLIRCIMInterpreter
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRLinalgDialect
)

mlir_check_all_link_libraries(cim-run)
