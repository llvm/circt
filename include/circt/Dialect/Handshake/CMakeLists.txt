add_circt_dialect(HandshakeOps handshake)
add_circt_doc(HandshakeOps -gen-op-doc HandshakeOps Dialects/)

set(LLVM_TARGET_DEFINITIONS HandshakeOps.td)
mlir_tablegen(HandshakeOps.inc -gen-rewriters)
add_public_tablegen_target(MLIRHandshakeRewritersIncGen)

set(LLVM_TARGET_DEFINITIONS HandshakeInterfaces.td)
mlir_tablegen(HandshakeInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(HandshakeInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRHandshakeInterfacesIncGen)

#set(LLVM_TARGET_DEFINITIONS HandshakeOps.td)
#mlir_tablegen(HandshakeOps.h.inc - gen - op - decls)
#mlir_tablegen(HandshakeOps.cpp.inc - gen - op - defs)
#mlir_tablegen(HandshakeOps.inc - gen - rewriters)
#add_public_tablegen_target(MLIRHandshakeOpsIncGen)
