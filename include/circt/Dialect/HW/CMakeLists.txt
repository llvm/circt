add_circt_dialect(HW hw)
add_circt_doc(HW -gen-dialect-doc HW Dialects/)

set(LLVM_TARGET_DEFINITIONS HW.td)

mlir_tablegen(HWAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(HWAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRHWAttrIncGen)
add_dependencies(circt-headers MLIRHWAttrIncGen)

mlir_tablegen(HWEnums.h.inc -gen-enum-decls)
mlir_tablegen(HWEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHWEnumsIncGen)
add_dependencies(circt-headers MLIRHWEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTHWTransformsIncGen)
add_circt_doc(Passes -gen-pass-doc HWPasses ./)

set(LLVM_TARGET_DEFINITIONS HWOpInterfaces.td)
mlir_tablegen(HWOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(HWOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(CIRCTHWOpInterfacesIncGen)
add_dependencies(circt-headers CIRCTHWOpInterfacesIncGen)
