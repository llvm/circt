file(GLOB globbed *.cpp)

add_mlir_library(SPTFIRToMLIR
  ${globbed}
  
  ADDITIONAL_HEADER_DIRS
  DEPENDS
  intrinsics_gen
  )

add_dependencies(SPTFIRToMLIR MLIRFIRRTL)
target_link_libraries(SPTFIRToMLIR MLIRFIRRTL)
