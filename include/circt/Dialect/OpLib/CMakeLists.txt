add_circt_dialect(OpLib oplib)
add_circt_dialect_doc(OpLib oplib)

set(LLVM_TARGET_DEFINITIONS OpLib.td)

mlir_tablegen(OpLibAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(OpLibAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(CIRCTOpLibAttributesIncGen)
add_dependencies(circt-headers CIRCTOpLibAttributesIncGen)
