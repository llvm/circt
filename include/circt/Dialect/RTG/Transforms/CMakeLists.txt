set(LLVM_TARGET_DEFINITIONS RTGPasses.td)
mlir_tablegen(RTGPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTRTGTransformsIncGen)
