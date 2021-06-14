file(GLOB globbed *.cpp)

add_circt_translation_library(CIRCTExportVerilog
  ${globbed}
  
  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  CIRCTComb
  CIRCTHW
  CIRCTSupport
  CIRCTSV
  MLIRTranslation
  )
