add_circt_dialect(SSP ssp)
add_circt_dialect_doc(SSP ssp)

set(LLVM_TARGET_DEFINITIONS SSP.td)

mlir_tablegen(SSPAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(SSPAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(CIRCTSSPAttributesIncGen)
add_dependencies(circt-headers CIRCTSSPAttributesIncGen)
