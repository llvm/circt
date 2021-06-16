file(GLOB globbed *.cpp)
add_circt_translation_library(CIRCTHIRToVerilog
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/circt/Target/HIRToVerilog
  ${PROJECT_SOURCE_DIR}/lib/Target/HIRToVerilog/

  LINK_LIBS PUBLIC
  CIRCTHIR
  MLIRTranslation
  )
