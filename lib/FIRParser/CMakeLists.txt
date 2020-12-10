file(GLOB globbed *.cpp)

add_circt_library(CIRCTFIRParser
  ${globbed}
  
  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  MLIRFIRRTL
  MLIRTranslation
  )
