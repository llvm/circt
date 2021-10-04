file(GLOB globbed *.cpp)

add_circt_translation_library(CIRCTExportVerilog
  ${globbed}
  
  ADDITIONAL_HEADER_DIRS

  DEPENDS
  CIRCTTranslationIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSupport
  CIRCTSV
  MLIRTranslation
  )
