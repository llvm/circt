set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name LECTransforms)
mlir_tablegen(LECTransforms.capi.h.inc -gen-pass-capi-header --prefix LECTransforms)
mlir_tablegen(LECTransforms.capi.cpp.inc -gen-pass-capi-impl --prefix LECTransforms)
add_public_tablegen_target(CIRCTLECTransformsIncGen)

add_circt_doc(Passes CIRCTLECPasses -gen-pass-doc)
