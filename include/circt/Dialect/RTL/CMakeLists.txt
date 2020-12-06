add_mlir_dialect(RTL rtl)
add_mlir_doc(RTL -gen-op-doc rtl Dialect/)

set(LLVM_TARGET_DEFINITIONS RTL.td)
mlir_tablegen(RTLEnums.h.inc -gen-enum-decls)
mlir_tablegen(RTLEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRRTLEnumsIncGen)
