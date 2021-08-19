add_circt_dialect(HW hw)
add_circt_doc(HW -gen-dialect-doc HW Dialects/)

set(LLVM_TARGET_DEFINITIONS HW.td)

mlir_tablegen(HWEnums.h.inc -gen-enum-decls)
mlir_tablegen(HWEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHWEnumsIncGen)
add_dependencies(circt-headers MLIRHWEnumsIncGen)

mlir_tablegen(HWStructs.h.inc -gen-struct-attr-decls)
mlir_tablegen(HWStructs.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRHWStructsIncGen)
add_dependencies(circt-headers MLIRHWStructsIncGen)

set(LLVM_TARGET_DEFINITIONS HWInterfaces.td)

mlir_tablegen(HWOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(HWOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRHWOpInterfacesIncGen)
add_dependencies(circt-headers MLIRHWOpInterfacesIncGen)
