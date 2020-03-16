add_mlir_library(SPTFIRToMLIR
  FIRToMLIR.cpp
  
  ADDITIONAL_HEADER_DIRS
  DEPENDS
  intrinsics_gen
  )

add_dependencies(SPTFIRToMLIR MLIRFIRRTL)
target_link_libraries(SPTFIRToMLIR MLIRFIRRTL)
