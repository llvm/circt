add_mlir_library(MLIRLLHDTargetVerilog
  VerilogPrinter.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Target
  )
target_link_libraries(MLIRLLHDTargetVerilog
  PUBLIC
  MLIRLLHD
  )
