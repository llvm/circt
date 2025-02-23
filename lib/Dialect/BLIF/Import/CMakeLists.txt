
add_circt_translation_library(CIRCTImportBLIFFile
  BLIFLexer.cpp
  BLIFParser.cpp

  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  CIRCTBLIF
  MLIRTranslateLib
  )
