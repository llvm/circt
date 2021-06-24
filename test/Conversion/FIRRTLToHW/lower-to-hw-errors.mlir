// RUN: circt-opt -lower-firrtl-to-hw %s -verify-diagnostics
// The firrtl.circuit should be removed, the main module name moved to an
// attribute on the module.
// CHECK-NOT: firrtl.circuit

// We should get a large header boilerplate.
// CHECK:   sv.ifdef.procedural "RANDOMIZE_GARBAGE_ASSIGN"  {
// CHECK-NEXT:   sv.verbatim "`define RANDOMIZE"
// CHECK-NEXT:  }
firrtl.circuit "InvalidBundle" {

  // https://github.com/llvm/circt/issues/593
  firrtl.module @InvalidBundle() {
    // expected-error @+1 {{unsupported type}}
    %0 = firrtl.invalidvalue : !firrtl.bundle<inp_d: uint<14>>
  }

  // expected-error @+1 {{unexpected operation 'func' in a firrtl.circuit}}
  func private @UnknownFunction() {
    return
  }
  firrtl.module @unprocessedAnnotations() {
  // expected-warning @+1 {{Unprocessed annotations}}
    %io_cpu_flush.wire = firrtl.wire {annotations = [{class = "firrtl.transforms.RemainingAnnotation"}]} : !firrtl.uint<1>
  }
}
