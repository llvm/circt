add_circt_dialect(Calyx calyx)
add_circt_doc(Calyx  -gen-dialect-doc Calyx Dialects/)

set(LLVM_TARGET_DEFINITIONS Calyx.td)

mlir_tablegen(CalyxEnums.h.inc -gen-enum-decls)
mlir_tablegen(CalyxEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRCalyxEnumsIncGen)
add_dependencies(circt-headers MLIRCalyxEnumsIncGen)

mlir_tablegen(CalyxAttrs.h.inc -gen-struct-attr-decls)
mlir_tablegen(CalyxAttrs.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRCalyxAttrsIncGen)
add_dependencies(circt-headers MLIRCalyxAttrsIncGen)
