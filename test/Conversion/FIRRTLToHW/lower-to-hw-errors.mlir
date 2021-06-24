// RUN: circt-opt -lower-firrtl-to-hw -verify-diagnostics -split-input-file %s 
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
}

// -----

firrtl.circuit "unprocessedAnnotations" {
  firrtl.module @bar(in %io_cpu_flush: !firrtl.uint<1>) {
  }
  firrtl.module @unprocessedAnnotations(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                            in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<2>) {
    // expected-warning @+1 {{Unprocessed annotations}}
    %1 = firrtl.wire {annotations = [{class = "firrtl.transforms.RemainingAnnotation"}]} : !firrtl.uint<1>
    // expected-warning @+1 {{Unprocessed annotations}}
    %2 = firrtl.node %1 {annotations = [{class = "firrtl.transforms.RemainingAnnotation"}]} : !firrtl.uint<1>
    // expected-warning @+1 {{Unprocessed annotations}}
    %3 = firrtl.reg %clock {annotations = [{class = "firrtl.transforms.RemainingAnnotation"}]} : (!firrtl.clock) -> !firrtl.uint<1>
    // expected-warning @+1 {{Unprocessed annotations}}
    %4 = firrtl.regreset %clock, %reset, %1 {annotations = [{class = "firrtl.transforms.RemainingAnnotation"}]} : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // expected-warning @+1 {{Unprocessed annotations}}
    %_M_read, %_M_rw, %_M_write = firrtl.mem Undefined {depth = 12 : i64, name = "_M", portNames = ["read", "rw", "write"], readLatency = 0 : i32, writeLatency = 1 : i32, annotations = [{class = "firrtl.transforms.RemainingAnnotation"}]} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, wmode: uint<1>, rdata flip: sint<42>, wdata: sint<42>, wmask: uint<1>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>
    // expected-warning @+1 {{Unprocessed annotations}}
    %5 = firrtl.instance @bar {name = "fetch", portNames=["io_cpu_flush"], annotations = [{class = "firrtl.transforms.RemainingAnnotation"}] } : !firrtl.uint<1>
  }
}
