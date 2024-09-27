set(LLVM_TARGET_DEFINITIONS Passes.td)
if(CIRCT_YOSYS_INTEGRATION_ENABLED)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Conversion -DCIRCT_YOSYS_INTEGRATION_ENABLED)
else()
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Conversion)
endif()

mlir_tablegen(Conversion.capi.h.inc -gen-pass-capi-header --prefix Conversion)
mlir_tablegen(Conversion.capi.cpp.inc -gen-pass-capi-impl --prefix Conversion)
add_public_tablegen_target(CIRCTConversionPassIncGen)

add_circt_doc(Passes CIRCTConversionPasses -gen-pass-doc)
