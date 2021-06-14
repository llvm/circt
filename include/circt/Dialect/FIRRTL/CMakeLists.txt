add_circt_dialect(FIRRTL firrtl FIRRTL)
add_circt_doc(FIRRTL -gen-dialect-doc FIRRTL Dialects/)

set(LLVM_TARGET_DEFINITIONS FIRRTL.td)
mlir_tablegen(FIRRTLEnums.h.inc -gen-enum-decls)
mlir_tablegen(FIRRTLEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(FIRRTLAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(FIRRTLAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRFIRRTLEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTFIRRTLTransformsIncGen)
add_circt_doc(Passes -gen-pass-doc FIRRTLPasses ./)

set(LLVM_TARGET_DEFINITIONS FIRRTLCanonicalization.td)
mlir_tablegen(FIRRTLCanonicalization.h.inc -gen-rewriters)
add_public_tablegen_target(CIRCTFIRRTLCanonicalizationIncGen)
