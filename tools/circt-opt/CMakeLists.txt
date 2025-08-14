get_property(dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY CIRCT_CONVERSION_LIBS)

set(LLVM_LINK_COMPONENTS
  Support
)

add_circt_tool(circt-opt
  circt-opt.cpp

  DEPENDS
  SUPPORT_PLUGINS
)

llvm_update_compile_flags(circt-opt)

target_link_libraries(circt-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}

  CIRCTAnalysisTestPasses
  CIRCTBMCTransforms
  CIRCTExportChiselInterface
  CIRCTExportVerilog
  CIRCTLECTransforms
  CIRCTCoreToFSM
  CIRCTTransforms

  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRSCFDialect
  MLIREmitCDialect
  MLIRFuncInlinerExtension
  MLIRVectorDialect
  MLIRIndexDialect
)

export_executable_symbols_for_plugins(circt-opt)
