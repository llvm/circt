set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CIRCTConversion)
mlir_tablegen(Conversion.capi.h.inc -gen-pass-capi-header --prefix CIRCTConversion)
mlir_tablegen(Conversion.capi.cpp.inc -gen-pass-capi-impl --prefix CIRCTConversion)
add_public_tablegen_target(CIRCTConversionPassIncGen)

add_circt_doc(Passes CIRCTConversionPasses -gen-pass-doc)
