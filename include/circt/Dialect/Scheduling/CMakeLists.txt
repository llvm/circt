add_circt_dialect(Scheduling sched)
add_circt_doc(Scheduling -gen-dialect-doc Scheduling Dialects/)

set(LLVM_TARGET_DEFINITIONS Scheduling.td)
mlir_tablegen(SchedulingAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(SchedulingAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRSchedulingAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS SchedulingInterfaces.td)
mlir_tablegen(SchedulingInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(SchedulingInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRSchedulingInterfacesIncGen)

add_dependencies(circt-headers
  MLIRSchedulingAttributesIncGen
  MLIRSchedulingInterfacesIncGen
  )
