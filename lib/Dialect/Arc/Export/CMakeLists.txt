add_circt_translation_library(CIRCTArcExportModelInfo
  ExportModelInfo.cpp

  LINK_LIBS PUBLIC
  CIRCTArc
  MLIRTranslateLib
)
