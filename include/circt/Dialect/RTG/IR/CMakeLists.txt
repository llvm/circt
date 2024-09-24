add_circt_dialect(RTG rtg)

set(LLVM_TARGET_DEFINITIONS RTG.td)

add_circt_dialect_doc(RTG rtg)

set(LLVM_TARGET_DEFINITIONS RTGInterfaces.td)
mlir_tablegen(RTGInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(RTGInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(CIRCTRTGInterfacesIncGen)
add_dependencies(circt-headers CIRCTRTGInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS ../Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTRTGTransformsIncGen)
