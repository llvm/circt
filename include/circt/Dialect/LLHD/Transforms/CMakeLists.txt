set(LLVM_TARGET_DEFINITIONS LLHDPasses.td)
mlir_tablegen(LLHDPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTLLHDTransformsIncGen)

add_circt_doc(LLHDPasses LLHDPasses -gen-pass-doc)
