file(GLOB globbed *.cpp)

add_mlir_library(SPTFIRParser
  ${globbed}
  
  ADDITIONAL_HEADER_DIRS
  DEPENDS
  intrinsics_gen
  )

add_dependencies(SPTFIRParser MLIRFIRRTL)
target_link_libraries(SPTFIRParser MLIRFIRRTL)
