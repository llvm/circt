add_circt_translation_library(CIRCTExportBLIF
  BLIFEmitter.cpp

  LINK_LIBS PUBLIC
  CIRCTBLIF
  MLIRTranslateLib
)
