add_circt_dialect(FSM fsm)
add_circt_doc(FSM -gen-dialect-doc FSM Dialects/)

set(LLVM_TARGET_DEFINITIONS FSMPasses.td)
mlir_tablegen(FSMPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTFSMTransformsIncGen)
add_circt_doc(Passes -gen-pass-doc FSMPasses ./)
