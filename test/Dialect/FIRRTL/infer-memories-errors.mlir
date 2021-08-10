// RUN: circt-opt -verify-diagnostics -pass-pipeline='firrtl.circuit(firrtl.module(firrtl-infer-memories))'  %s

firrtl.circuit "NoInferredEnables" {
firrtl.module @NoInferredEnables(in %p: !firrtl.uint<1>, in %addr: !firrtl.uint<4>, in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %v: !firrtl.uint<32>) {
  %ram = firrtl.seqmem Undefined  : !firrtl.cmemory<uint<32>, 16>
  %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
  %r = firrtl.regreset %clock, %reset, %c0_ui4  : !firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>
  // expected-warning @+1 {{memory port is never enabled}}
  %ramport = firrtl.memoryport Read %ram, %r, %clock  : (!firrtl.cmemory<uint<32>, 16>, !firrtl.uint<4>, !firrtl.clock) -> !firrtl.uint<32>
  firrtl.connect %v, %ramport : !firrtl.uint<32>, !firrtl.uint<32>
}
}
