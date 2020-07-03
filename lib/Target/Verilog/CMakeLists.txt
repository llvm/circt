add_mlir_library(MLIRLLHDTargetVerilog
  TranslateToVerilog.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/circt/Target/Verilog

  LINK_LIBS PUBLIC
  MLIRLLHD
  )
