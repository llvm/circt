add_circt_dialect(HW rtl)
add_circt_doc(HW -gen-dialect-doc HW Dialects/)

set(LLVM_TARGET_DEFINITIONS HW.td)
mlir_tablegen(HWEnums.h.inc -gen-enum-decls)
mlir_tablegen(HWEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHWEnumsIncGen)
