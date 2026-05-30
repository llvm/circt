add_llvm_library(CIRCTStandalonePlugin
  MODULE BUILDTREE_ONLY
  circt-standalone-plugin.cpp

  DEPENDS
  CIRCTStandalone
  PLUGIN_TOOL
  circt-opt

  LINK_LIBS
  CIRCTStandalone
)

target_include_directories(CIRCTStandalonePlugin
  PRIVATE
  "${CIRCT_STANDALONE_BINARY_DIR}/include"
)
