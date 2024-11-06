add_circt_dialect(RTGTest rtgtest)

set(LLVM_TARGET_DEFINITIONS RTGTest.td)
mlir_tablegen(RTGTestEnums.h.inc -gen-enum-decls)
mlir_tablegen(RTGTestEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(RTGTestEnumsIncGen)

add_circt_dialect_doc(RTGTest rtgtest)
