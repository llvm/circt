add_circt_dialect_library(CIRCTRTGTransforms
  ContextPass.cpp
  ElaborationPass.cpp
  
  DEPENDS
  CIRCTRTGTransformsIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PRIVATE
  CIRCTRTGDialect
  MLIRIR
)

