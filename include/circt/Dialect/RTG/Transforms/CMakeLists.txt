set(LLVM_TARGET_DEFINITIONS RTGPasses.td)
mlir_tablegen(RTGPasses.h.inc -gen-pass-decls -name RTG)
mlir_tablegen(RTGPasses.capi.h.inc -gen-pass-capi-header -prefix RTG)
mlir_tablegen(RTGPasses.capi.cpp.inc -gen-pass-capi-impl -prefix RTG)
add_public_tablegen_target(CIRCTRTGTransformsIncGen)

# Generate Pass documentation.
add_circt_doc(RTGPasses RTGPasses -gen-pass-doc)
