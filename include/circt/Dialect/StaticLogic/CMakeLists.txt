add_circt_dialect(StaticLogic staticlogic)
add_circt_doc(StaticLogic -gen-dialect-doc StaticLogic Dialects/)

set(LLVM_TARGET_DEFINITIONS StaticLogicPasses.td)
mlir_tablegen(StaticLogicPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRStaticLogicPassesIncGen)
add_dependencies(circt-headers MLIRStaticLogicPassesIncGen)
add_circt_doc(StaticLogicPasses -gen-pass-doc StaticLogicPasses ./)
