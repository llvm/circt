set(LLVM_TARGET_DEFINITIONS TranslationPasses.td)
mlir_tablegen(TranslationPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTTranslationIncGen)
add_circt_doc(TranslationPasses -gen-pass-doc TranslationPasses ./)
