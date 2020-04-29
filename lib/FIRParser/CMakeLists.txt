file(GLOB globbed *.cpp)

add_mlir_library(CIRTFIRParser
  ${globbed}
  
  ADDITIONAL_HEADER_DIRS
  )

add_dependencies(CIRTFIRParser MLIRFIRRTL)
target_link_libraries(CIRTFIRParser MLIRFIRRTL)
