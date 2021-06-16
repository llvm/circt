set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTTransformsIncGen)
add_circt_doc(Passes -gen-pass-doc Passes ./)
