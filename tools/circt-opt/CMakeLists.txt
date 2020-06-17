set(LIBS
  MLIRFIRRTL
  MLIRRTL

  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRStandardOps
  MLIRTransforms
  LLVMSupport
)

add_llvm_tool(circt-opt
 circt-opt.cpp
)

llvm_update_compile_flags(circt-opt)
target_link_libraries(circt-opt PRIVATE ${LIBS} ${targets_to_link})

