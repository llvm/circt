file(GLOB globbed *.cpp)

add_circt_translation_library(CIRCTImportFIRFile
  ${globbed}
  
  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  MLIRTranslation
  )
