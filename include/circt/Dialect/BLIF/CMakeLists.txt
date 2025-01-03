add_circt_dialect(BLIF blif)

set(LLVM_TARGET_DEFINITIONS BLIF.td)
mlir_tablegen(BLIFEnums.h.inc -gen-enum-decls)
mlir_tablegen(BLIFEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTBLIFEnumsIncGen)
add_dependencies(circt-headers CIRCTBLIFEnumsIncGen)
