add_circt_dialect(FIRRTL firrtl FIRRTL)
add_circt_doc(FIRRTL -gen-op-doc firrtl Dialect/)

set(LLVM_TARGET_DEFINITIONS FIRRTL.td)
mlir_tablegen(FIRRTLEnums.h.inc -gen-enum-decls)
mlir_tablegen(FIRRTLEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(FIRRTLPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRFIRRTLEnumsIncGen)
