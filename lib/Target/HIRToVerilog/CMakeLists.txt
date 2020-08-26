add_mlir_library(MLIRHIRToVerilog
  HIRToVerilog.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/circt/Target/HIRToVerilog

  LINK_LIBS PUBLIC
  MLIRHIR
  )
