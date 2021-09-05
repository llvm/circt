add_circt_dialect(Calyx calyx)
add_circt_doc(Calyx -gen-dialect-doc Calyx Dialects/)

set(LLVM_TARGET_DEFINITIONS Calyx.td)

mlir_tablegen(CalyxEnums.h.inc -gen-enum-decls)
mlir_tablegen(CalyxEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRCalyxEnumsIncGen)
add_dependencies(circt-headers MLIRCalyxEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS CalyxAttributes.td)
mlir_tablegen(CalyxAttributes.h.inc -gen-struct-attr-decls)
mlir_tablegen(CalyxAttributes.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRCalyxAttributesIncGen)
add_dependencies(circt-headers MLIRCalyxAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS CalyxInterfaces.td)
mlir_tablegen(CalyxInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(CalyxInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRCalyxInterfacesIncGen)
add_dependencies(circt-headers MLIRCalyxInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS CalyxPasses.td)
mlir_tablegen(CalyxPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTCalyxTransformsIncGen)
add_circt_doc(CalyxPasses -gen-pass-doc CalyxPasses ./)
