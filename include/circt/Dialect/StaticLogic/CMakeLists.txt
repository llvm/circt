add_mlir_dialect(StaticLogic staticlogic)
add_mlir_doc(StaticLogic -gen-op-doc staticlogic Dialect/)

set(LLVM_TARGET_DEFINITIONS StaticLogic.td)
