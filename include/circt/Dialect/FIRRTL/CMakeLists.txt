add_circt_dialect(FIRRTL firrtl FIRRTL)
add_circt_doc(FIRRTL -gen-op-doc firrtl Dialect/)

set(LLVM_TARGET_DEFINITIONS FIRRTL.td)
mlir_tablegen(FIRRTLEnums.h.inc -gen-enum-decls)
mlir_tablegen(FIRRTLEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRFIRRTLEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTFIRRTLTransformsIncGen)
add_circt_doc(Passes -gen-pass-doc FIRRTLPasses ./)
