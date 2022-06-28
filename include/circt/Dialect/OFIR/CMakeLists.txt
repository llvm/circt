# OFIR
add_circt_dialect(OFIR ofir OFIR)
add_circt_dialect_doc(OFIR ofir)

set(LLVM_TARGET_DEFINITIONS OFIR.td)
#mlir_tablegen(OFIREnums.h.inc -gen-enum-decls)
#mlir_tablegen(OFIREnums.cpp.inc -gen-enum-defs)
mlir_tablegen(OFIRAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(OFIRAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(CIRCTOFIREnumsIncGen)

set(LLVM_TARGET_DEFINITIONS OFIRCanonicalization.td)
mlir_tablegen(OFIRCanonicalization.h.inc -gen-rewriters)
add_public_tablegen_target(CIRCTOFIRCanonicalizationIncGen)
