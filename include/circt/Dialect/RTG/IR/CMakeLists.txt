add_circt_dialect(RTG rtg)

set(LLVM_TARGET_DEFINITIONS RTG.td)

add_circt_dialect_doc(RTG rtg)

set(LLVM_TARGET_DEFINITIONS RTGInterfaces.td)
mlir_tablegen(RTGOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(RTGOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(CIRCTRTGOpInterfacesIncGen)
add_dependencies(circt-headers CIRCTRTGOpInterfacesIncGen)

mlir_tablegen(RTGTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(RTGTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(CIRCTRTGTypeInterfacesIncGen)
add_dependencies(circt-headers CIRCTRTGTypeInterfacesIncGen)
