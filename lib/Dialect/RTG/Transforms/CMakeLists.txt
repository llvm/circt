add_circt_dialect_library(CIRCTRTGTransforms
  ElaborationPass.cpp
  EmbedValidationValuesPass.cpp
  EmitRTGISAAssemblyPass.cpp
  InlineSequencesPass.cpp
  LinearScanRegisterAllocationPass.cpp
  LowerUniqueLabelsPass.cpp
  LowerValidateToLabelsPass.cpp
  MemoryAllocationPass.cpp
  UniqueValidateOpsPass.cpp

  DEPENDS
  CIRCTRTGTransformsIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PRIVATE
  CIRCTRTGDialect
  CIRCTSupport
  MLIRArithDialect
  MLIRIndexDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
)

