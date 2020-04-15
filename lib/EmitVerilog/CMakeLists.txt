file(GLOB globbed *.cpp)

add_mlir_library(SPTEmitVerilog
  ${globbed}
  
  ADDITIONAL_HEADER_DIRS
  )

add_dependencies(SPTEmitVerilog MLIRFIRRTL)
target_link_libraries(SPTEmitVerilog MLIRFIRRTL)
