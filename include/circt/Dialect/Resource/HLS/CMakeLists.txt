add_circt_dialect_doc(HLS hls)

set(LLVM_TARGET_DEFINITIONS HLS.td)
mlir_tablegen(HLS.h.inc -gen-op-decls)
mlir_tablegen(HLS.cpp.inc -gen-op-defs)
mlir_tablegen(HLSDialect.h.inc -gen-dialect-decls -dialect=hls)
mlir_tablegen(HLSDialect.cpp.inc -gen-dialect-defs -dialect=hls)
add_public_tablegen_target(CIRCTHLSIncGen)