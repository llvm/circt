add_circt_dialect(Comb comb)
add_circt_dialect_doc(Comb comb)

set(LLVM_TARGET_DEFINITIONS Comb.td)
mlir_tablegen(CombEnums.h.inc -gen-enum-decls)
mlir_tablegen(CombEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRCombEnumsIncGen)
