add_mlir_library(MLIRCIMInterpreter
  Interpreter.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/cim/Interpreter

  LINK_LIBS PUBLIC
  MLIRCIM
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSupport
)

# Executing cim ops means calling the real runtime, not a mock: every byte
# the IR says crosses an address-space boundary really crosses cimrt's.
target_link_libraries(MLIRCIMInterpreter PUBLIC cimrt)
cim_apply_build_flags(MLIRCIMInterpreter)
