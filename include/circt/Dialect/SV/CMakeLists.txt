add_mlir_dialect(SV sv)
add_mlir_doc(SV -gen-op-doc sv Dialect/)

set(LLVM_TARGET_DEFINITIONS SV.td)

mlir_tablegen(SVEnums.h.inc -gen-enum-decls)
mlir_tablegen(SVEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRSVEnumsIncGen)
add_dependencies(mlir-headers MLIRSVEnumsIncGen)

mlir_tablegen(SVStructs.h.inc -gen-struct-attr-decls)
mlir_tablegen(SVStructs.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRSVStructsIncGen)
add_dependencies(mlir-headers MLIRSVStructsIncGen)
