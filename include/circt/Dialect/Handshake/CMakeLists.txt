add_circt_dialect(Handshake handshake)
add_circt_doc(Handshake -gen-op-doc Handshake Dialects/)
add_circt_interface(HandshakeInterfaces)

set(LLVM_TARGET_DEFINITIONS Handshake.td)

mlir_tablegen(HandshakeEnums.h.inc -gen-enum-decls)
mlir_tablegen(HandshakeEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHandshakeEnumsIncGen)
add_dependencies(circt-headers MLIRHandshakeEnumsIncGen)

mlir_tablegen(HandshakeAttrs.h.inc -gen-struct-attr-decls)
mlir_tablegen(HandshakeAttrs.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRHandshakeAttrsIncGen)
add_dependencies(circt-headers MLIRHandshakeAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS HandshakeCanonicalization.td)
mlir_tablegen(HandshakeCanonicalization.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRHandshakeCanonicalizationIncGen)

add_subdirectory(Transforms)
