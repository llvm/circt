add_circt_dialect(SystemC systemc)
add_circt_dialect_doc(SystemC systemc)

set(LLVM_TARGET_DEFINITIONS SystemC.td)

mlir_tablegen(SystemCEnums.h.inc -gen-enum-decls)
mlir_tablegen(SystemCEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTSystemCEnumsIncGen)
add_dependencies(circt-headers CIRCTSystemCEnumsIncGen)

mlir_tablegen(SystemCAttributes.h.inc -gen-attrdef-decls
  -attrdefs-dialect SystemCDialect)
mlir_tablegen(SystemCAttributes.cpp.inc -gen-attrdef-defs
  -attrdefs-dialect SystemCDialect)
add_public_tablegen_target(CIRCTSystemCAttributesIncGen)
add_dependencies(circt-headers CIRCTSystemCAttributesIncGen)
