add_circt_dialect(StaticLogic staticlogic)
add_circt_doc(StaticLogic -gen-dialect-doc StaticLogic Dialects/)

set(LLVM_TARGET_DEFINITIONS StaticLogicAttributes.td)
mlir_tablegen(StaticLogicAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(StaticLogicAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRStaticLogicAttrIncGen)
add_dependencies(circt-headers MLIRStaticLogicAttrIncGen)
