add_circt_dialect_library(CIRCTRTGTransforms
  ElaborationPass.cpp

  DEPENDS
  CIRCTRTGTransformsIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PRIVATE
  CIRCTRTGDialect
  MLIRArithDialect
  MLIRIR
  MLIRPass
)

