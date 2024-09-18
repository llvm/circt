add_circt_dialect(RTG rtg)
add_circt_dialect_doc(RTG rtg)

set(LLVM_TARGET_DEFINITIONS RTG.td)
mlir_tablegen(RTGEnums.h.inc -gen-enum-decls)
mlir_tablegen(RTGEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRRTGEnumsIncGen)

add_dependencies(circt-headers MLIRRTGIncGen MLIRRTGEnumsIncGen)
