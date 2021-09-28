add_circt_dialect(LLHD llhd)
add_circt_doc(LLHD -gen-dialect-doc LLHD Dialects/)

set(LLVM_TARGET_DEFINITIONS LLHD.td)
mlir_tablegen(LLHDEnums.h.inc -gen-enum-decls)
mlir_tablegen(LLHDEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTLLHDEnumsIncGen)

mlir_tablegen(LLHDAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(LLHDAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(CIRCTLLHDAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS LLHDCanonicalization.td)
mlir_tablegen(LLHDCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(CIRCTLLHDCanonicalizationIncGen)
