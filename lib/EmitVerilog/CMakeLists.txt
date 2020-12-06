file(GLOB globbed *.cpp)

add_mlir_library(CIRCTEmitVerilog
  ${globbed}
  
  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  MLIRFIRRTL
  MLIRRTL
  MLIRSV
  MLIRTranslation
  )
