add_mlir_dialect(FIRRTL firrtl FIRRTL)

set(LLVM_TARGET_DEFINITIONS FIRRTL.td)
mlir_tablegen(FIRRTLEnums.h.inc -gen-enum-decls)
mlir_tablegen(FIRRTLEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRFIRRTLEnumsIncGen)


