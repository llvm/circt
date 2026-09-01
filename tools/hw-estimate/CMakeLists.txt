# circt/tools/hw-estimate/CMakeLists.txt
set(LLVM_LINK_COMPONENTS Support)

add_circt_tool(hw-estimate
  hw-estimate.cpp
)

target_link_libraries(hw-estimate
  PRIVATE
  CIRCTHW
  CIRCTComb
  CIRCTSeq
  MLIRIR
  MLIRParser
  MLIRSupport
)
