add_circt_dialect(AIG aig)
add_circt_dialect_doc(AIG aig)

set(LLVM_TARGET_DEFINITIONS AIGPasses.td)
mlir_tablegen(AIGPasses.h.inc -gen-pass-decls)
mlir_tablegen(AIGPassesEnums.h.inc -gen-enum-decls)
mlir_tablegen(AIGPassesEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTAIGPassesIncGen)
add_circt_doc(AIGPasses AIGPasses -gen-pass-doc)
