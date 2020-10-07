set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(circt-opt
 circt-opt.cpp
)
llvm_update_compile_flags(circt-opt)
target_link_libraries(circt-opt
  PRIVATE
  MLIRFIRRTL
  MLIRStaticLogicOps
  MLIRHandshakeOps
  MLIRRTL
  MLIRSV
  MLIRStandardToHandshake
  MLIRStandardToStaticLogic
  MLIRHandshakeToFIRRTL
  MLIRLLHD
  MLIRLLHDTransforms
  MLIRLLHDToLLVM

  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRStandard
  MLIRTransforms
  MLIRLLVMIR
  )

