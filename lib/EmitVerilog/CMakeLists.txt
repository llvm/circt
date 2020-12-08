file(GLOB globbed *.cpp)

add_circt_library(CIRCTEmitVerilog
  ${globbed}
  
  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  MLIRFIRRTL
  MLIRRTL
  MLIRSV
  MLIRTranslation
  )
