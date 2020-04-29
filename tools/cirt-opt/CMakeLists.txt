set(LIBS
  MLIRFIRRTL

  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRStandardOps
  MLIRTransforms
  LLVMSupport
)

add_llvm_tool(cirt-opt
 cirt-opt.cpp
)

llvm_update_compile_flags(cirt-opt)
target_link_libraries(cirt-opt PRIVATE ${LIBS} ${targets_to_link})

