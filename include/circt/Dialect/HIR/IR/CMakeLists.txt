add_circt_dialect(HIR hir)
add_circt_doc(HIR -gen-op-doc HIR HIR/)
set(LLVM_TARGET_DEFINITIONS HIR.td)

set(LLVM_TARGET_DEFINITIONS HIROpInterfaces.td)
mlir_tablegen(HIROpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(HIROpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRHIRInterfacesIncGen)
