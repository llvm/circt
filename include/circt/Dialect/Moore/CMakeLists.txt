add_circt_dialect(Moore moore)
add_circt_doc(Moore -gen-dialect-doc Moore Dialects/)

set(LLVM_TARGET_DEFINITIONS Moore.td)

mlir_tablegen(MooreEnums.h.inc -gen-enum-decls)
mlir_tablegen(MooreEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTMooreEnumsIncGen)
add_dependencies(circt-headers CIRCTMooreEnumsIncGen)

mlir_tablegen(MooreStructs.h.inc -gen-struct-attr-decls)
mlir_tablegen(MooreStructs.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(CIRCTMooreStructsIncGen)
add_dependencies(circt-headers CIRCTMooreStructsIncGen)

mlir_tablegen(MooreAttributes.h.inc -gen-attrdef-decls
  -attrdefs-dialect MooreDialect)
mlir_tablegen(MooreAttributes.cpp.inc -gen-attrdef-defs
  -attrdefs-dialect MooreDialect)
add_public_tablegen_target(CIRCTMooreAttributesIncGen)
add_dependencies(circt-headers CIRCTMooreAttributesIncGen)
