include(SlangCompilerOptions)

add_circt_unittest(CIRCTImportVerilogTests
  CaptureAnalysisTest.cpp
)

target_include_directories(CIRCTImportVerilogTests
  PRIVATE
  ${CIRCT_MAIN_SRC_DIR}
)

target_link_libraries(CIRCTImportVerilogTests
  PRIVATE
  CIRCTImportVerilog
  LLVMSupport
  slang_slang
)
