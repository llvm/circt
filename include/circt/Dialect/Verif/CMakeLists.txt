add_circt_dialect(Verif verif)
add_circt_doc(VerifOps Dialects/VerifOps -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS Verif.td)
mlir_tablegen(VerifEnums.h.inc -gen-enum-decls)
mlir_tablegen(VerifEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTVerifEnumsIncGen)
add_dependencies(circt-headers CIRCTVerifEnumsIncGen)
add_circt_interface(VerifOpInterfaces)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTVerifTransformsIncGen)
