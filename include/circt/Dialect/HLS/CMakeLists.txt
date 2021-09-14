add_circt_dialect(HLS hls)
add_circt_doc(HLS -gen-dialect-doc HLS Dialects/)

set(LLVM_TARGET_DEFINITIONS HLS.td)
mlir_tablegen(HLSAttributes.h.inc -gen-struct-attr-decls)
mlir_tablegen(HLSAttributes.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(CIRCTHLSAttributesIncGen)

mlir_tablegen(HLSEnums.h.inc -gen-enum-decls)
mlir_tablegen(HLSEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTHLSEnumsIncGen)
add_dependencies(circt-headers CIRCTHLSEnumsIncGen)
