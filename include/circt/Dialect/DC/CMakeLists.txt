add_circt_dialect(DC dc)
add_circt_dialect_doc(DC dc)

set(LLVM_TARGET_DEFINITIONS DCPasses.td)
mlir_tablegen(DCPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTDCTransformsIncGen)
add_circt_doc(DCPasses DCPasses -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS DCCanonicalization.td)
mlir_tablegen(DCCanonicalization.h.inc -gen-rewriters)
add_public_tablegen_target(CIRCTDCCanonicalizationIncGen)
add_dependencies(circt-headers CIRCTDCCanonicalizationIncGen)
