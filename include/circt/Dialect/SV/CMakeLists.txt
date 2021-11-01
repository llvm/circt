add_circt_dialect(SV sv)
add_circt_doc(SV -gen-dialect-doc SV Dialects/)

set(LLVM_TARGET_DEFINITIONS SV.td)

mlir_tablegen(SVEnums.h.inc -gen-enum-decls)
mlir_tablegen(SVEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRSVEnumsIncGen)
add_dependencies(circt-headers MLIRSVEnumsIncGen)

mlir_tablegen(SVStructs.h.inc -gen-struct-attr-decls)
mlir_tablegen(SVStructs.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRSVStructsIncGen)
add_dependencies(circt-headers MLIRSVStructsIncGen)

mlir_tablegen(SVAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect sv)
mlir_tablegen(SVAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect sv)
add_public_tablegen_target(MLIRSVAttrIncGen)
add_circt_doc(SV -gen-attrdef-doc SVAttributes Dialects/)

set(LLVM_TARGET_DEFINITIONS SVPasses.td)
mlir_tablegen(SVPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTSVTransformsIncGen)
add_circt_doc(SVPasses -gen-pass-doc SVPasses ./)
