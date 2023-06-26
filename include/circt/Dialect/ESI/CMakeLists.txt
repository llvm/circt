add_circt_dialect(ESI esi)
add_circt_dialect_doc(ESI esi)

set(LLVM_TARGET_DEFINITIONS ESI.td)
mlir_tablegen(ESIPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRESITransformsIncGen)
add_circt_doc(ESI ESIPasses -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS ESI.td)
mlir_tablegen(ESIEnums.h.inc -gen-enum-decls)
mlir_tablegen(ESIEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRESIEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS ESIInterfaces.td)
mlir_tablegen(ESIInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ESIInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRESIInterfacesIncGen)
add_dependencies(circt-headers MLIRESIInterfacesIncGen)

add_subdirectory(cosim)
