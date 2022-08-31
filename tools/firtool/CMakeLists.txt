set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(firtool
  firtool.cpp
)
llvm_update_compile_flags(firtool)
target_link_libraries(firtool PRIVATE
  CIRCTExportChiselInterface
  CIRCTExportVerilog
  CIRCTImportFIRFile
  CIRCTFIRRTLToHW
  CIRCTFIRRTLTransforms
  CIRCTHWTransforms
  CIRCTSeq
  CIRCTSeqTransforms
  CIRCTSVTransforms
  CIRCTTransforms

  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRFuncDialect
  MLIRTransforms
)
