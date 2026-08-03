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

# Warnings in the dialect are ours to fix; without this the -Werror build
# skips this library entirely and real defects (a dead `unsigned < 0` check,
# a signed/unsigned comparison) sit in it unnoticed.
target_link_libraries(MLIRCIM PUBLIC cim_build_flags)
