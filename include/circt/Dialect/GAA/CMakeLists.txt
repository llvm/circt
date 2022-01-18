add_circt_dialect(GAA gaa)
add_circt_doc(GAA -gen-dialect-doc GAA Dialects/)
add_circt_doc(GAAPasses -gen-pass-doc GAAPasses ./)

set(LLVM_TARGET_DEFINITIONS GAA.td)

mlir_tablegen(GAAAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(GAAAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(CIRCTGAAAttrsIncGen)
add_dependencies(circt-headers CIRCTGAAAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS GAAOpInterfaces.td)
mlir_tablegen(GAAOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(GAAOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(CIRCTGAAOpInterfacesIncGen)
add_dependencies(circt-headers CIRCTGAAOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS GAAPasses.td)
mlir_tablegen(GAAPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTGAATransformsIncGen)

add_dependencies(circt-headers MLIRGAAIncGen)