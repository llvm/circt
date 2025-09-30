add_circt_dialect(Datapath datapath)
add_circt_dialect_doc(Datapath datapath)

set(LLVM_TARGET_DEFINITIONS Datapath.td)
set(LLVM_TARGET_DEFINITIONS DatapathPasses.td)
mlir_tablegen(DatapathPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTDatapathTransformsIncGen)

# Generate Pass documentation.
add_circt_doc(Passes DatapathPasses -gen-pass-doc)
