add_subdirectory(FIRRTLToLLHD)
add_subdirectory(LLHDToLLVM)
add_subdirectory(RTLToLLHD)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(CIRCTConversionPassIncGen)

add_circt_doc(Passes -gen-pass-doc CIRCTConversionPasses ./)
