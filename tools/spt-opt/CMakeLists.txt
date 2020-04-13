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

add_llvm_tool(spt-opt
 spt-opt.cpp
)

llvm_update_compile_flags(spt-opt)
target_link_libraries(spt-opt PRIVATE ${LIBS} ${targets_to_link})

