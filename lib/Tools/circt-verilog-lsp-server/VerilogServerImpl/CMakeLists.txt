# Include the Slang compiler options
include(SlangCompilerOptions)

add_circt_library(CIRCTVerilogLspServerImpl
  VerilogServer.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/circt/Tools/crct-verilog-lsp-server

  DEPENDS
  slang_slang

  LINK_LIBS PUBLIC
  MLIRLspServerSupportLib
  CIRCTVerilogLspServerUtils
  PRIVATE
  slang_slang
)

# HACK: When `OBJECT` argument is passed to `llvm_add_library()`,
# `COMPILE_DEFINITIONS` are not correctly inherited. For that reason,
# we manually set it here.
if(TARGET obj.CIRCTVerilogLspServerImpl AND TARGET Boost::headers)
  get_target_property(compile_defs Boost::headers INTERFACE_COMPILE_DEFINITIONS)
  target_compile_definitions(obj.CIRCTVerilogLspServerImpl PRIVATE ${compile_defs})
endif()