add_circt_dialect(RTGTest rtgtest)

set(LLVM_TARGET_DEFINITIONS RTGTest.td)

add_circt_dialect_doc(RTGTest rtgtest)

set(LLVM_TARGET_DEFINITIONS RTGTestAttributes.td)
mlir_tablegen(RTGTestEnums.h.inc -gen-enum-decls)
mlir_tablegen(RTGTestEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTRTGTestEnumsIncGen)

mlir_tablegen(RTGTestAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(RTGTestAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(CIRCTRTGTestAttributeIncGen)
