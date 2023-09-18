set(LLVM_LINK_COMPONENTS
  Support
)

add_circt_tool(firtool
  firtool.cpp
)
llvm_update_compile_flags(firtool)
target_link_libraries(firtool PRIVATE
  CIRCTExportVerilog
  CIRCTImportFIRFile
  CIRCTFIRRTLTransforms
  CIRCTHWTransforms
  CIRCTOMTransforms
  CIRCTSeq
  CIRCTSVTransforms
  CIRCTTransforms
  CIRCTFirtool
  CIRCTTargetDebugInfo

  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRFuncDialect
  MLIRTransforms
)
