add_subdirectory(FIRToMLIR)

file(GLOB globbed *.cpp)
add_mlir_dialect_library(MLIRFIRRTL
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${SPT_MAIN_INCLUDE_DIR}/spt/Dialect/FIRRTL
  )
add_dependencies(MLIRFIRRTL MLIRFIRRTLIncGen MLIRFIRRTLEnumsIncGen LLVMSupport)
target_link_libraries(MLIRFIRRTL LLVMSupport)
