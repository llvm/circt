add_mlir_dialect(ESI esi)
add_mlir_doc(ESI -gen-op-doc esi Dialect/)

set(LLVM_TARGET_DEFINITIONS ESI.td)
mlir_tablegen(ESIAttrs.h.inc -gen-struct-attr-decls)
mlir_tablegen(ESIAttrs.cpp.inc -gen-struct-attr-defs)
mlir_tablegen(ESIPasses.h.inc -gen-pass-decls)

add_public_tablegen_target(MLIRESIEnumsIncGen)

add_subdirectory(cosim)
