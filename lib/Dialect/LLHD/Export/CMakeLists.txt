add_circt_translation_library(CIRCTExportLLHD
  TranslateToVerilog.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/circt/Target/Verilog

  LINK_LIBS PUBLIC
  CIRCTLLHD
  CIRCTComb
  CIRCTHW
  MLIRMemRef
  MLIRTranslation
  )
