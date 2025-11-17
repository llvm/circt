include(SlangCompilerOptions)

add_circt_translation_library(CIRCTImportVerilog
  AssertionExpr.cpp
  Expressions.cpp
  FormatStrings.cpp
  HierarchicalNames.cpp
  ImportVerilog.cpp
  Statements.cpp
  Structure.cpp
  Types.cpp
  TimingControls.cpp

  DEPENDS
  slang_slang

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTDebug
  CIRCTHW
  CIRCTLLHD
  CIRCTLLHDTransforms
  CIRCTLTL
  CIRCTMoore
  CIRCTMooreToCore
  CIRCTMooreTransforms
  CIRCTSeq
  CIRCTSeqTransforms
  CIRCTSupport
  CIRCTTransforms
  CIRCTVerif
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRTranslateLib
  PRIVATE
  slang_slang
)
