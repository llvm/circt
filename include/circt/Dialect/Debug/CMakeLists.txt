add_circt_dialect(Debug dbg)
add_circt_doc(DebugOps Dialects/DebugOps -gen-op-doc)
add_circt_doc(DebugTypes Dialects/DebugTypes -gen-typedef-doc -dialect dbg)

add_dependencies(circt-headers MLIRDebugIncGen)
