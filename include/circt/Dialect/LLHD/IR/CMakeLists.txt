add_mlir_dialect(LLHD llhd)
add_mlir_doc(LLHD -gen-op-doc llhd Dialect/)

set(LLVM_TARGET_DEFINITIONS LLHD.td)
mlir_tablegen(LLHDEnums.h.inc -gen-enum-decls)
mlir_tablegen(LLHDEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRLLHDEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS LLHDCanonicalization.td)
mlir_tablegen(LLHDCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(MLIRLLHDCanonicalizationIncGen)
