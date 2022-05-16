add_circt_dialect(ESI esi)
add_circt_dialect_doc(ESI esi)

set(LLVM_TARGET_DEFINITIONS ESI.td)
mlir_tablegen(ESIAttrs.h.inc -gen-struct-attr-decls)
mlir_tablegen(ESIAttrs.cpp.inc -gen-struct-attr-defs)
mlir_tablegen(ESIPasses.h.inc -gen-pass-decls)

add_circt_doc(ESI ESIPasses -gen-pass-doc)

add_public_tablegen_target(MLIRESIEnumsIncGen)

add_subdirectory(cosim)
