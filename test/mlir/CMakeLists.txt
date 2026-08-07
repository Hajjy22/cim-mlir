# Tests that need the MLIR layer. Kept separate from cim-unit-tests so the
# core suite stays buildable and runnable without a toolchain.
add_executable(cim-mlir-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/../unit/test_main.cpp
  pipeline_e2e_test.cpp
  cost_report_e2e_test.cpp
  schedule_e2e_test.cpp
  legalize_precision_e2e_test.cpp
)

target_include_directories(cim-mlir-tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../unit
)

target_link_libraries(cim-mlir-tests PRIVATE
  MLIRCIM
  MLIRCIMTransforms
  MLIRCIMInterpreter
  MLIRParser
  MLIRPass
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  cim_build_flags
)

llvm_update_compile_flags(cim-mlir-tests)

# Run from the repo root so test/targets/tiny-4x4.yaml resolves.
add_test(NAME cim-mlir-tests
  COMMAND cim-mlir-tests
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
