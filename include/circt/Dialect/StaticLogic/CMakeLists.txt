add_circt_dialect(StaticLogic staticlogic)
add_circt_doc(StaticLogic -gen-op-doc staticlogic Dialects/)

set(LLVM_TARGET_DEFINITIONS StaticLogic.td)
