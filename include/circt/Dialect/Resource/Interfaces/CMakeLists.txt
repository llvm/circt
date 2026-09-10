set(LLVM_TARGET_DEFINITIONS MemoryInterface.td)
mlir_tablegen(MemoryOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(MemoryOpInterface.cpp.inc -gen-op-interface-defs)
mlir_tablegen(MemoryDialectInterface.h.inc -gen-dialect-interface-decls)
add_public_tablegen_target(MemoryInterfaceIncGen)


