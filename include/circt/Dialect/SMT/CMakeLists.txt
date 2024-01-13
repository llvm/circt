add_circt_dialect(SMT smt)
add_circt_doc(SMTOps Dialects/SMTOps -gen-op-doc)
add_circt_doc(SMTTypes Dialects/SMTTypes -gen-typedef-doc -dialect smt)

set(LLVM_TARGET_DEFINITIONS SMT.td)
mlir_tablegen(SMTEnums.h.inc -gen-enum-decls)
mlir_tablegen(SMTEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTSMTEnumsIncGen)
add_dependencies(circt-headers CIRCTSMTEnumsIncGen)
