// RUN: circt-opt -verify-diagnostics -pass-pipeline='firrtl.circuit(firrtl.module(firrtl-lower-chirrtl))'  %s

firrtl.circuit "NoInferredEnables" {
firrtl.module @NoInferredEnables(in %p: !firrtl.uint<1>, in %addr: !firrtl.uint<4>, in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %v: !firrtl.uint<32>) {
  %ram = chirrtl.seqmem Undefined  : !chirrtl.cmemory<uint<32>, 16>
  %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
  %r = firrtl.regreset %clock, %reset, %c0_ui4  : !firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>
  // expected-warning @+1 {{memory port is never enabled}}
  %ramport_data, %ramport_port = chirrtl.memoryport Read %ram {name = "ramport"} : (!chirrtl.cmemory<uint<32>, 16>) -> (!firrtl.uint<32>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %ramport_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint<4>, !firrtl.clock

  firrtl.connect %v, %ramport_data : !firrtl.uint<32>, !firrtl.uint<32>
}

firrtl.module @Bitindex(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io: !firrtl.bundle<addr: uint<32>, wdata: uint<8>>) {
  %0 = firrtl.subfield %io(1) : (!firrtl.bundle<addr: uint<32>, wdata: uint<8>>) -> !firrtl.uint<8>
  %1 = firrtl.subfield %io(0) : (!firrtl.bundle<addr: uint<32>, wdata: uint<8>>) -> !firrtl.uint<32>
  %mem = chirrtl.seqmem interesting_name Undefined  : !chirrtl.cmemory<uint<8>, 1024>
  %MPORT_data, %MPORT_port = chirrtl.memoryport Write %mem  {name = "MPORT"} : (!chirrtl.cmemory<uint<8>, 1024>) -> (!firrtl.uint<8>, !chirrtl.cmemoryport)
  %2 = firrtl.bits %MPORT_data 0 to 0 : (!firrtl.uint<8>) -> !firrtl.uint<1>
  %3 = firrtl.bits %1 9 to 0 : (!firrtl.uint<32>) -> !firrtl.uint<10>
  %_T = firrtl.node interesting_name %3  : !firrtl.uint<10>
  chirrtl.memoryport.access %MPORT_port[%_T], %clock : !chirrtl.cmemoryport, !firrtl.uint<10>, !firrtl.clock
  %4 = firrtl.tail %0, 7 : (!firrtl.uint<8>) -> !firrtl.uint<1>
  // expected-error @+1 {{cannot use bit-index to write CHIRRTL memory}}
  firrtl.strictconnect %2, %4 : !firrtl.uint<1>
}
}
