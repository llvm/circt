add_circt_dialect(CalyxLib calyxlib)
add_circt_dialect_doc(CalyxLib calyxlib)

set(LLVM_TARGET_DEFINITIONS CalyxLib.td)

mlir_tablegen(CalyxLibAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(CalyxLibAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(CIRCTCalyxLibAttributesIncGen)
add_dependencies(circt-headers CIRCTCalyxLibAttributesIncGen)
