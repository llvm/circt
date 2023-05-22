add_circt_dialect(Pipeline pipeline)
add_circt_dialect_doc(Pipeline pipeline)

set(LLVM_TARGET_DEFINITIONS Pipeline.td)

set(LLVM_TARGET_DEFINITIONS PipelinePasses.td)
mlir_tablegen(PipelinePasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTPipelineTransformsIncGen)
add_circt_doc(PipelinePasses PipelinePasses -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS PipelineInterfaces.td)
mlir_tablegen(PipelineInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(PipelineInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(CIRCTPipelineInterfacesIncGen)
add_dependencies(circt-headers CIRCTPipelineInterfacesIncGen)
