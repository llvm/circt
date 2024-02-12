add_circt_translation_library(CIRCTExportArc
  ModelInfoExport.cpp

  LINK_LIBS PUBLIC
  CIRCTArc
  MLIRTranslateLib
)
