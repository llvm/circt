set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Analysis)
add_public_tablegen_target(CIRCTAnalysisPassIncGen)

add_circt_doc(Passes -gen-pass-doc CIRCTAnalysisPasses ./)
