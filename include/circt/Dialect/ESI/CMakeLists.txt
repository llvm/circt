add_mlir_dialect(ESI esi)
add_mlir_doc(ESI -gen-op-doc esi Dialect/)

set(LLVM_TARGET_DEFINITIONS ESI.td)
mlir_tablegen(ESIEnums.h.inc -gen-enum-decls)
mlir_tablegen(ESIEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(MLIRESIEnumsIncGen)
