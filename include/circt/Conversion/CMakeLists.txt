add_subdirectory(FIRRTLToLLHD)
add_subdirectory(LLHDToLLVM)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(CIRCTConversionPassIncGen)

add_mlir_doc(Passes -gen-pass-doc CIRCTConversionPasses ./)
