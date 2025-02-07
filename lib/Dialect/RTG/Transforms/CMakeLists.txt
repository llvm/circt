add_circt_dialect_library(CIRCTRTGTransforms
  ElaborationPass.cpp
  EmitRTGISAAssemblyPass.cpp
  LinearScanRegisterAllocationPass.cpp

  DEPENDS
  CIRCTRTGTransformsIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PRIVATE
  CIRCTRTGDialect
  CIRCTSupport
  MLIRIndexDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
)

