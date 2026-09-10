
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CIRCTTransforms)
mlir_tablegen(Transforms.capi.h.inc -gen-pass-capi-header --prefix CIRCTTransforms)
mlir_tablegen(Transforms.capi.cpp.inc -gen-pass-capi-impl --prefix CIRCTTransforms)
add_public_tablegen_target(CIRCTTransformsPassIncGen)

add_circt_doc(Passes CIRCTGeneralPasses -gen-pass-doc)
