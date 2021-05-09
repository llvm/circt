add_circt_dialect(Scheduling sched)
add_circt_doc(Scheduling -gen-dialect-doc Scheduling Dialects/)

set(LLVM_TARGET_DEFINITIONS Scheduling.td)
mlir_tablegen(SchedulingAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(SchedulingAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRSchedulingExtraIncGen)

add_dependencies(circt-headers MLIRSchedulingExtraIncGen)
