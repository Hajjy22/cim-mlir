add_mlir_dialect_library(MLIRCIM
  IR/CIMDialect.cpp
  IR/CIMTypes.cpp
  IR/CIMOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/cim/Dialect

  DEPENDS
  MLIRCIMOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
)
