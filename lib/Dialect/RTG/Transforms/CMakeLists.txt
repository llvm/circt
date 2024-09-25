add_circt_dialect_library(CIRCTRTGTransforms
  ExportElf.cpp

  DEPENDS
  CIRCTRTGTransformsIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PRIVATE
  CIRCTRTGDialect
  MLIRIR
)

