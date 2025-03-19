add_circt_dialect_library(CIRCTRTGTransforms
  ElaborationPass.cpp
  EmitRTGISAAssemblyPass.cpp
  InlineSequencesPass.cpp
  LinearScanRegisterAllocationPass.cpp
  LowerUniqueLabelsPass.cpp

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

