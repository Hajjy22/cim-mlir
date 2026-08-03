add_mlir_dialect_library(MLIRCIMTransforms
  CIMDetect.cpp
  CIMPartition.cpp
  CIMPlacement.cpp
  CIMSchedule.cpp
  CIMInsertTransfers.cpp
  CIMLegalizePrecision.cpp
  CIMLowerToTarget.cpp
  CIMCostReport.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/cim/Transforms

  DEPENDS
  MLIRCIMTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRCIM
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)

# The passes delegate their actual algorithms to the MLIR-free core.
target_link_libraries(MLIRCIMTransforms PUBLIC cimPlacement cimTarget)
