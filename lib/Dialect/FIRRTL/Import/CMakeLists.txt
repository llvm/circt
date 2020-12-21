file(GLOB globbed *.cpp)

add_circt_translation_library(CIRCTImportFIRRTL
  ${globbed}
  
  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  MLIRFIRRTL
  MLIRTranslation
  )
