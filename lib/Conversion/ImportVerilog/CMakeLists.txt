include(SlangCompilerOptions)

add_circt_translation_library(CIRCTImportVerilog
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
  CIRCTDebug
  CIRCTHW
  CIRCTMoore
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRTranslateLib
  PRIVATE
  slang_slang
)

# HACK: When `OBJECT` argument is passed to `llvm_add_library()`,
# `COMPILE_DEFINITIONS` are not correctly inherited. For that reason,
# we manually set it here.
if(TARGET obj.CIRCTImportVerilog AND TARGET Boost::headers)
  get_target_property(compile_defs Boost::headers INTERFACE_COMPILE_DEFINITIONS)
  target_compile_definitions(obj.CIRCTImportVerilog PRIVATE ${compile_defs})
endif()
