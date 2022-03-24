add_circt_dialect(CHALK chalk CHALK)
add_circt_doc(CHALK -gen-dialect-doc CHALK Dialects/)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTCHALKTransformsIncGen)
add_circt_doc(Passes -gen-pass-doc CHALKPasses ./)
