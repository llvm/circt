add_mlir_dialect(LLHDOps llhd)
add_mlir_doc(LLHDOps -gen-op-doc llhd Dialect/)

set(LLVM_TARGET_DEFINITIONS LLHDOps.td)
mlir_tablegen(LLHDOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(LLHDOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRLLHDEnumsIncGen)
