add_circt_dialect(Sim sim)
add_circt_doc(SimOps Dialects/SimOps -gen-op-doc)
add_circt_doc(SimTypes Dialects/SimTypes -gen-typedef-doc -dialect sim)

set(LLVM_TARGET_DEFINITIONS Sim.td)

mlir_tablegen(SimAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=sim)
mlir_tablegen(SimAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=sim)
add_public_tablegen_target(MLIRSimAttributesIncGen)
add_dependencies(circt-headers MLIRSimAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS SimPasses.td)
mlir_tablegen(SimPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTSimTransformsIncGen)
add_dependencies(circt-headers CIRCTSimTransformsIncGen)
