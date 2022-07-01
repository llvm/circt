add_circt_dialect(ESI esi)
add_circt_dialect_doc(ESI esi)

set(LLVM_TARGET_DEFINITIONS ESI.td)
mlir_tablegen(ESIPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRESITransformsIncGen)
add_circt_doc(ESI ESIPasses -gen-pass-doc)

add_subdirectory(cosim)
