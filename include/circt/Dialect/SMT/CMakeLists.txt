add_circt_dialect(SMT smt)
add_circt_doc(SMT Dialects/SMTOps -gen-op-doc)
add_circt_doc(SMT Dialects/SMTTypes -gen-typedef-doc -dialect smt)

set(LLVM_TARGET_DEFINITIONS SMT.td)

mlir_tablegen(SMTAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(SMTAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(CIRCTSMTAttrIncGen)
add_dependencies(circt-headers CIRCTSMTAttrIncGen)

mlir_tablegen(SMTEnums.h.inc -gen-enum-decls)
mlir_tablegen(SMTEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTSMTEnumsIncGen)
add_dependencies(circt-headers CIRCTSMTEnumsIncGen)
