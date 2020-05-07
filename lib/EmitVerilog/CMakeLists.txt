file(GLOB globbed *.cpp)

add_mlir_library(CIRTEmitVerilog
  ${globbed}
  
  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  MLIRFIRRTL
  )
