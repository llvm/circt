set(LIBS
  MLIRFIRRTL
  CIRTEmitVerilog
  CIRTFIRParser

  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRStandardOps
  MLIRTransforms
  MLIRTranslation
  LLVMSupport
)

add_llvm_tool(firtool
 firtool.cpp
)

llvm_update_compile_flags(firtool)
target_link_libraries(firtool PRIVATE ${LIBS} ${targets_to_link})

