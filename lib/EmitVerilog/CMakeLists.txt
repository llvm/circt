file(GLOB globbed *.cpp)

add_mlir_library(CIRTEmitVerilog
  ${globbed}
  
  ADDITIONAL_HEADER_DIRS
  )

add_dependencies(CIRTEmitVerilog MLIRFIRRTL)
target_link_libraries(CIRTEmitVerilog MLIRFIRRTL)
