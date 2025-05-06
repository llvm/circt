set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name BMCTransforms)
mlir_tablegen(BMCTransforms.capi.h.inc -gen-pass-capi-header --prefix BMCTransforms)
mlir_tablegen(BMCTransforms.capi.cpp.inc -gen-pass-capi-impl --prefix BMCTransforms)
add_public_tablegen_target(CIRCTBMCTransformsIncGen)

add_circt_doc(Passes CIRCTBMCPasses -gen-pass-doc)
