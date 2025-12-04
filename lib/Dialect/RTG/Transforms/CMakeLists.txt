add_circt_dialect_library(CIRCTRTGTransforms
  ElaborationPass.cpp
  EmbedValidationValuesPass.cpp
  EmitRTGISAAssemblyPass.cpp
  InlineSequencesPass.cpp
  InsertTestToFileMappingPass.cpp
  LinearScanRegisterAllocationPass.cpp
  LowerUniqueLabelsPass.cpp
  LowerValidateToLabelsPass.cpp
  MaterializeConstraintsPass.cpp
  MemoryAllocationPass.cpp
  PrintTestNamesPass.cpp
  RTGPassPipelines.cpp
  SimpleTestInlinerPass.cpp
  UniqueValidateOpsPass.cpp

  DEPENDS
  CIRCTRTGTransformsIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PRIVATE
  CIRCTRTGDialect
  CIRCTEmit
  CIRCTSupport
  MLIRArithDialect
  MLIRIndexDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
)

