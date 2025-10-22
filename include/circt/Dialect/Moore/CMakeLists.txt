add_circt_dialect(Moore moore)
add_circt_doc(MooreOps Dialects/MooreOps -gen-op-doc)
add_circt_doc(MooreTypes Dialects/MooreTypes -gen-typedef-doc -dialect moore)

set(LLVM_TARGET_DEFINITIONS Moore.td)

mlir_tablegen(MooreEnums.h.inc -gen-enum-decls)
mlir_tablegen(MooreEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTMooreEnumsIncGen)
add_dependencies(circt-headers CIRCTMooreEnumsIncGen)

mlir_tablegen(MooreAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect moore)
mlir_tablegen(MooreAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect moore)
add_public_tablegen_target(CIRCTMooreAttributesIncGen)
add_dependencies(circt-headers CIRCTMooreAttributesIncGen)

add_dependencies(CIRCTMooreAttributesIncGen CIRCTMooreEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS MoorePasses.td)
mlir_tablegen(MoorePasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTMooreTransformsIncGen)
