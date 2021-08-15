// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl.module(firrtl-infer-memories))'  %s | FileCheck %s

firrtl.circuit "Empty" {

// Memories with no ports should be deleted.
firrtl.module @Empty(in %clock: !firrtl.clock) {
  %ram = firrtl.combmem : !firrtl.cmemory<uint<1>, 2>
}
// CHECK:      firrtl.module @Empty(in %clock: !firrtl.clock) {
// CHECK-NEXT: }

// Unused ports should be deleted.
firrtl.module @UnusedMemPort(in %clock: !firrtl.clock, in %addr : !firrtl.uint<1>) {
  %ram = firrtl.combmem : !firrtl.cmemory<vector<uint<1>, 2>, 2>
  // This port should be deleted.
  %port0, %port0_port = firrtl.memoryport Infer %ram : (!firrtl.cmemory<vector<uint<1>, 2>, 2>) -> (!firrtl.vector<uint<1>, 2>, !firrtl.cmemoryport)
  firrtl.memoryport.access %port0_port[%addr], %clock : !firrtl.cmemoryport, !firrtl.uint<1>, !firrtl.clock
  // Subindexing a port should not count as a "use".
  %port1, %port1_port = firrtl.memoryport Infer %ram : (!firrtl.cmemory<vector<uint<1>, 2>, 2>) -> (!firrtl.vector<uint<1>, 2>, !firrtl.cmemoryport)
  firrtl.memoryport.access %port1_port[%addr], %clock : !firrtl.cmemoryport, !firrtl.uint<1>, !firrtl.clock
  %0 = firrtl.subindex %port1[1] : !firrtl.vector<uint<1>, 2>
}
// CHECK:      firrtl.module @UnusedMemPort(in %clock: !firrtl.clock, in %addr: !firrtl.uint<1>) {
// CHECK-NEXT: }

firrtl.module @InferRead(in %cond: !firrtl.uint<1>, in %clock: !firrtl.clock, in %addr: !firrtl.uint<8>, out %out : !firrtl.uint<1>, in %vec : !firrtl.vector<uint<1>, 2>) {
  // CHECK: %ram_ramport = firrtl.mem Undefined {depth = 256 : i64, name = "ram", portNames = ["ramport"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<8>, en: uint<1>, clk: clock, data flip: uint<1>>
  // CHECK: [[ADDR:%.*]] = firrtl.subfield %ram_ramport(0)
  // CHECK: firrtl.connect [[ADDR]], %invalid_ui8
  // CHECK: [[EN:%.*]] = firrtl.subfield %ram_ramport(1)
  // CHECK: firrtl.connect [[EN]], %c0_ui1
  // CHECK: [[CLOCK:%.*]] = firrtl.subfield %ram_ramport(2)
  // CHECK: firrtl.connect [[CLOCK]], %invalid_clock
  // CHECK: [[DATA:%.*]] = firrtl.subfield %ram_ramport(3)
  %ram = firrtl.combmem : !firrtl.cmemory<uint<1>, 256>
  %ramport, %ramport_port = firrtl.memoryport Infer %ram : (!firrtl.cmemory<uint<1>, 256>) -> (!firrtl.uint<1>, !firrtl.cmemoryport)

  // CHECK: firrtl.when %cond {
  // CHECK:   firrtl.connect [[ADDR]], %addr
  // CHECK:   firrtl.connect [[EN]], %c1_ui1
  // CHECK:   firrtl.connect [[CLOCK]], %clock
  // CHECK: }
  firrtl.when %cond {
    firrtl.memoryport.access %ramport_port[%addr], %clock : !firrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
  }

  // CHECK: %node = firrtl.node [[DATA]]
  %node = firrtl.node %ramport : !firrtl.uint<1>

  // CHECK: firrtl.connect %out, [[DATA]]
  firrtl.connect %out, %ramport : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.partialconnect %out, [[DATA]]
  firrtl.partialconnect %out, %ramport : !firrtl.uint<1>, !firrtl.uint<1>

  // TODO: How do you get FileCheck to accept "[[[DATA]]]"?
  // CHECK: firrtl.subaccess %vec{{\[}}[[DATA]]{{\]}} : !firrtl.vector<uint<1>, 2>, !firrtl.uint<1>
  firrtl.subaccess %vec[%ramport] : !firrtl.vector<uint<1>, 2>, !firrtl.uint<1>
}

firrtl.module @InferWrite(in %cond: !firrtl.uint<1>, in %clock: !firrtl.clock, in %addr: !firrtl.uint<8>, in %in : !firrtl.uint<1>) {
  // CHECK: %ram_ramport = firrtl.mem Undefined {depth = 256 : i64, name = "ram", portNames = ["ramport"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<8>, en: uint<1>, clk: clock, data: uint<1>, mask: uint<1>>
  // CHECK: [[ADDR:%.*]] = firrtl.subfield %ram_ramport(0)
  // CHECK: firrtl.connect [[ADDR]], %invalid_ui8
  // CHECK: [[EN:%.*]] = firrtl.subfield %ram_ramport(1)
  // CHECK: firrtl.connect [[EN]], %c0_ui1
  // CHECK: [[CLOCK:%.*]] = firrtl.subfield %ram_ramport(2)
  // CHECK: firrtl.connect [[CLOCK]], %invalid_clock
  // CHECK: [[DATA:%.*]] = firrtl.subfield %ram_ramport(3)
  // CHECK: firrtl.connect [[DATA]], %invalid_ui1
  // CHECK: [[MASK:%.*]] = firrtl.subfield %ram_ramport(4)
  // CHECK: firrtl.connect [[MASK]], %invalid_ui1
  %ram = firrtl.combmem : !firrtl.cmemory<uint<1>, 256>
  %ramport, %ramport_port = firrtl.memoryport Infer %ram : (!firrtl.cmemory<uint<1>, 256>) -> (!firrtl.uint<1>, !firrtl.cmemoryport)

  // CHECK: firrtl.when %cond {
  // CHECK:   firrtl.connect [[ADDR]], %addr
  // CHECK:   firrtl.connect [[EN]], %c1_ui1
  // CHECK:   firrtl.connect [[CLOCK]], %clock
  // CHECK:   firrtl.connect [[MASK]], %c0_ui1
  // CHECK: }
  firrtl.when %cond {
    firrtl.memoryport.access %ramport_port[%addr], %clock : !firrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
  }

  // CHECK: firrtl.connect [[MASK]], %c1_ui1
  // CHECK: firrtl.connect [[DATA]], %in
  firrtl.connect %ramport, %in : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.connect [[MASK]], %c1_ui1
  // CHECK: firrtl.partialconnect [[DATA]], %in
  firrtl.partialconnect %ramport, %in : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @InferReadWrite(in %clock: !firrtl.clock, in %addr: !firrtl.uint<8>, in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
  // CHECK: %ram_ramport = firrtl.mem Undefined {depth = 256 : i64, name = "ram", portNames = ["ramport"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<8>, en: uint<1>, clk: clock, rdata flip: uint<1>, wmode: uint<1>, wdata: uint<1>, wmask: uint<1>>
  // CHECK: [[ADDR:%.*]] = firrtl.subfield %ram_ramport(0)
  // CHECK: firrtl.connect [[ADDR]], %invalid_ui8
  // CHECK: [[EN:%.*]] = firrtl.subfield %ram_ramport(1)
  // CHECK: firrtl.connect [[EN]], %c0_ui1
  // CHECK: [[CLOCK:%.*]] = firrtl.subfield %ram_ramport(2)
  // CHECK: firrtl.connect [[CLOCK]], %invalid_clock
  // CHECK: [[RDATA:%.*]] = firrtl.subfield %ram_ramport(3)
  // CHECK: [[WMODE:%.*]] = firrtl.subfield %ram_ramport(4)
  // CHECK: firrtl.connect [[WMODE]], %c0_ui1
  // CHECK: [[WDATA:%.*]] = firrtl.subfield %ram_ramport(5)
  // CHECK: firrtl.connect [[WDATA]], %invalid_ui1
  // CHECK: [[WMASK:%.*]] = firrtl.subfield %ram_ramport(6)
  // CHECK: firrtl.connect [[WMASK]], %invalid_ui1
  %ram = firrtl.combmem : !firrtl.cmemory<uint<1>, 256>

  // CHECK: firrtl.connect [[ADDR]], %addr : !firrtl.uint<8>, !firrtl.uint<8>
  // CHECK: firrtl.connect [[EN]], %c1_ui1
  // CHECK: firrtl.connect [[CLOCK]], %clock
  // CHECK: firrtl.connect [[WMASK]], %c0_ui1
  %ramport, %ramport_port = firrtl.memoryport Read %ram : (!firrtl.cmemory<uint<1>, 256>) -> (!firrtl.uint<1>, !firrtl.cmemoryport)
  firrtl.memoryport.access %ramport_port[%addr], %clock : !firrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock

  // CHECK: firrtl.connect [[WMASK]], %c1_ui1
  // CHECK: firrtl.connect [[WMODE]], %c1_ui1
  // CHECK: firrtl.connect [[WDATA]], %in
  firrtl.connect %ramport, %in : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.connect %out, [[RDATA]] 
  firrtl.connect %out, %ramport : !firrtl.uint<1>, !firrtl.uint<1>
}

// Check that partial connect properly sets the write mask for the elements which are actually connected.
firrtl.module @PartialConnectWriteMask(in %clock: !firrtl.clock, in %addr: !firrtl.uint<8>, in %data : !firrtl.bundle<c: vector<uint<3>, 2>, a: uint<1>>) {
  // CHECK: %ram_ramport = firrtl.mem Undefined {depth = 256 : i64, name = "ram", portNames = ["ramport"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<8>, en: uint<1>, clk: clock, data: bundle<a: uint<1>, b: uint<2>, c: vector<uint<3>, 3>>, mask: bundle<a: uint<1>, b: uint<1>, c: vector<uint<1>, 3>>>
  // CHECK: [[DATA:%.*]] = firrtl.subfield %ram_ramport(3)
  // CHECK: [[MASK:%.*]] = firrtl.subfield %ram_ramport(4)
  // CHECK: [[A:%.*]] = firrtl.subfield [[MASK]](0)
  // CHECK: firrtl.connect [[A]], %invalid_ui1 
  // CHECK: [[B:%.*]] = firrtl.subfield [[MASK]](1)
  // CHECK: firrtl.connect [[B]], %invalid_ui1 
  // CHECK: [[C:%.*]] = firrtl.subfield [[MASK]](2)
  // CHECK: [[C_0:%.*]] = firrtl.subindex [[C]][0]
  // CHECK: firrtl.connect [[C_0]], %invalid_ui1 
  // CHECK: [[C_1:%.*]] = firrtl.subindex [[C]][1]
  // CHECK: firrtl.connect [[C_1]], %invalid_ui1 
  // CHECK: [[C_2:%.*]] = firrtl.subindex [[C]][2]
  // CHECK: firrtl.connect [[C_2]], %invalid_ui1 
  %ram = firrtl.combmem : !firrtl.cmemory<bundle<a: uint<1>, b: uint<2>, c: vector<uint<3>, 3>>, 256>

  // CHECK: [[A:%.*]] = firrtl.subfield [[MASK]](0)
  // CHECK: firrtl.connect [[A]], %c0_ui1 
  // CHECK: [[B:%.*]] = firrtl.subfield [[MASK]](1) : (!firrtl.bundle<a: uint<1>, b: uint<1>, c: vector<uint<1>, 3>>) -> !firrtl.uint<1>
  // CHECK: firrtl.connect [[B]], %c0_ui1 
  // CHECK: [[C:%.*]] = firrtl.subfield [[MASK]](2) : (!firrtl.bundle<a: uint<1>, b: uint<1>, c: vector<uint<1>, 3>>) -> !firrtl.vector<uint<1>, 3>
  // CHECK: [[C_0:%.*]] = firrtl.subindex [[C]][0] : !firrtl.vector<uint<1>, 3>
  // CHECK: firrtl.connect [[C_0]], %c0_ui1 
  // CHECK: [[C_1:%.*]] = firrtl.subindex [[C]][1] : !firrtl.vector<uint<1>, 3>
  // CHECK: firrtl.connect [[C_1]], %c0_ui1 
  // CHECK: [[C_2:%.*]] = firrtl.subindex [[C]][2] : !firrtl.vector<uint<1>, 3>
  // CHECK: firrtl.connect [[C_2]], %c0_ui1 
  %ramport, %ramport_port = firrtl.memoryport Infer %ram : (!firrtl.cmemory<bundle<a: uint<1>, b: uint<2>, c: vector<uint<3>, 3>>, 256>) -> (!firrtl.bundle<a: uint<1>, b: uint<2>, c: vector<uint<3>, 3>>, !firrtl.cmemoryport)
  firrtl.memoryport.access %ramport_port[%addr], %clock : !firrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock


  // CHECK: [[A:%.*]] = firrtl.subfield [[MASK]](0) : (!firrtl.bundle<a: uint<1>, b: uint<1>, c: vector<uint<1>, 3>>) -> !firrtl.uint<1>
  // CHECK: firrtl.connect [[A]], %c1_ui1 
  // CHECK: [[C:%.*]] = firrtl.subfield [[MASK]](2) : (!firrtl.bundle<a: uint<1>, b: uint<1>, c: vector<uint<1>, 3>>) -> !firrtl.vector<uint<1>, 3>
  // CHECK: [[C_0:%.*]] = firrtl.subindex [[C]][0] : !firrtl.vector<uint<1>, 3>
  // CHECK: firrtl.connect [[C_0]], %c1_ui1 
  // CHECK: [[C_1:%.*]] = firrtl.subindex [[C]][1] : !firrtl.vector<uint<1>, 3>
  // CHECK: firrtl.connect [[C_1]], %c1_ui1 
  // CHECK: firrtl.partialconnect [[DATA]], %data
  firrtl.partialconnect %ramport, %data : !firrtl.bundle<a: uint<1>, b: uint<2>, c: vector<uint<3>, 3>>, !firrtl.bundle<c: vector<uint<3>, 2>, a: uint<1>>
}

firrtl.module @WriteToSubfield(in %clock: !firrtl.clock, in %addr: !firrtl.uint<8>, in %value: !firrtl.uint<1>) {
  %ram = firrtl.combmem : !firrtl.cmemory<bundle<a: uint<1>, b: uint<1>>, 256>
  %ramport, %ramport_port = firrtl.memoryport Infer %ram : (!firrtl.cmemory<bundle<a: uint<1>, b: uint<1>>, 256>) -> (!firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.cmemoryport)
  firrtl.memoryport.access %ramport_port[%addr], %clock : !firrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock

  %ramport_b = firrtl.subfield %ramport(1) : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<1>
  // Check that only the subfield of the mask is written to.
  // CHECK: [[DATA:%.*]] = firrtl.subfield %ram_ramport(3)
  // CHECK: [[MASK:%.*]] = firrtl.subfield %ram_ramport(4)
  // CHECK: [[DATA_B:%.*]] = firrtl.subfield [[DATA]](1)
  // CHECK: [[MASK_B:%.*]] = firrtl.subfield [[MASK]](1)
  // CHECK: firrtl.connect [[MASK_B]], %c1_ui1
  // CHECK: firrtl.connect [[DATA_B]], %value
  firrtl.connect %ramport_b, %value : !firrtl.uint<1>, !firrtl.uint<1>
}

// Read and write from different subfields of the memory.  The memory as a
// whole should be inferred to read+write.
firrtl.module @ReadAndWriteToSubfield(in %clock: !firrtl.clock, in %addr: !firrtl.uint<8>, in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
  %ram = firrtl.combmem : !firrtl.cmemory<bundle<a: uint<1>, b: uint<1>>, 256>
  %ramport, %ramport_port = firrtl.memoryport Infer %ram : (!firrtl.cmemory<bundle<a: uint<1>, b:uint<1>>, 256>) -> (!firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.cmemoryport)
  firrtl.memoryport.access %ramport_port[%addr], %clock : !firrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock


  // CHECK: [[RDATA:%.*]] = firrtl.subfield %ram_ramport(3)
  // CHECK: [[WMODE:%.*]] = firrtl.subfield %ram_ramport(4)
  // CHECK: [[WDATA:%.*]] = firrtl.subfield %ram_ramport(5)
  // CHECK: [[WMASK:%.*]] = firrtl.subfield %ram_ramport(6)
  // CHECK: [[WDATA_A:%.*]] = firrtl.subfield [[WDATA]](0)
  // CHECK: [[WMASK_A:%.*]] = firrtl.subfield [[WMASK]](0)
  // CHECK: firrtl.connect [[WMASK_A]], %c1_ui1
  // CHECK: firrtl.connect [[WMODE]], %c1_ui1
  // CHECK: firrtl.connect [[WDATA_A]], %in
  %port_a = firrtl.subfield %ramport(0) : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<1>
  firrtl.connect %port_a, %in : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: [[RDATA_B:%.*]] = firrtl.subfield [[RDATA]](1) : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<1>
  // CHECK: firrtl.connect %out, [[RDATA_B]] : !firrtl.uint<1>, !firrtl.uint<1>
  %port_b = firrtl.subfield %ramport(1) : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<1>
  firrtl.connect %out, %port_b : !firrtl.uint<1>, !firrtl.uint<1>
}

// Check that ports are sorted in alphabetical order.
firrtl.module @SortedPorts(in %clock: !firrtl.clock, in %addr : !firrtl.uint<8>, out %out: !firrtl.uint<1>) {
  // CHECK: portNames = ["a", "b", "c"]
  %ram = firrtl.combmem : !firrtl.cmemory<vector<uint<1>, 2>, 256>
  %c, %c_port = firrtl.memoryport Read %ram : (!firrtl.cmemory<vector<uint<1>, 2>, 256>) -> (!firrtl.vector<uint<1>, 2>, !firrtl.cmemoryport)
  firrtl.memoryport.access %c_port[%addr], %clock : !firrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
  %a, %a_port = firrtl.memoryport Write %ram : (!firrtl.cmemory<vector<uint<1>, 2>, 256>) -> (!firrtl.vector<uint<1>, 2>, !firrtl.cmemoryport)
  firrtl.memoryport.access %a_port[%addr], %clock : !firrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
  %b, %b_port = firrtl.memoryport ReadWrite %ram : (!firrtl.cmemory<vector<uint<1>, 2>, 256>) -> (!firrtl.vector<uint<1>, 2>, !firrtl.cmemoryport)
  firrtl.memoryport.access %b_port[%addr], %clock : !firrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
}

// Check that annotations are preserved.
firrtl.module @Annotations(in %clock: !firrtl.clock, in %addr : !firrtl.uint<8>, out %out: !firrtl.uint<1>) {
  // CHECK: firrtl.mem Undefined 
  // CHECK-SAME: annotations = [{a = "a"}]
  // CHECK-SAME: portAnnotations = [
  // CHECK-SAME:   [{b = "b"}],
  // CHECK-SAME:   [{c = "c"}]
  // CHECK-SAME: ]
  // CHECK-SAME: portNames = ["port0", "port1"]
  %ram = firrtl.combmem {annotations = [{a = "a"}]} : !firrtl.cmemory<vector<uint<1>, 2>, 256>
  %port0, %port0_port = firrtl.memoryport Read %ram {annotations = [{b = "b"}]} : (!firrtl.cmemory<vector<uint<1>, 2>, 256>) -> (!firrtl.vector<uint<1>, 2>, !firrtl.cmemoryport)
  firrtl.memoryport.access %port0_port[%addr], %clock : !firrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
  %port1, %port1_port = firrtl.memoryport Read %ram {annotations = [{c = "c"}]} : (!firrtl.cmemory<vector<uint<1>, 2>, 256>) -> (!firrtl.vector<uint<1>, 2>, !firrtl.cmemoryport)
  firrtl.memoryport.access %port1_port[%addr], %clock : !firrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
}

// When the address is a wire, the enable should be inferred where the address
// is driven.
firrtl.module @EnableInference0(in %p: !firrtl.uint<1>, in %addr: !firrtl.uint<4>, in %clock: !firrtl.clock, out %v: !firrtl.uint<32>) {
  %w = firrtl.wire  : !firrtl.uint<4>
  // This connect should not count as "driving" a value.  If it accidentally
  // inserts an enable here, we will get a use-before-def error, so it is
  // enough of a check that this compiles.
  %invalid_ui4 = firrtl.invalidvalue : !firrtl.uint<4>
  firrtl.connect %w, %invalid_ui4 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: [[ADDR:%.*]] = firrtl.subfield %ram_ramport(0)
  // CHECK: [[EN:%.*]] = firrtl.subfield %ram_ramport(1)
  %ram = firrtl.seqmem Undefined  : !firrtl.cmemory<uint<32>, 16>
  %ramport, %ramport_port = firrtl.memoryport Read %ram : (!firrtl.cmemory<uint<32>, 16>) -> (!firrtl.uint<32>, !firrtl.cmemoryport)
  firrtl.memoryport.access %ramport_port[%w], %clock : !firrtl.cmemoryport, !firrtl.uint<4>, !firrtl.clock

  // CHECK: firrtl.when %p {
  firrtl.when %p  {
    // CHECK-NEXT: firrtl.connect [[EN]], %c1_ui1
    // CHECK-NEXT: firrtl.connect %w, %addr
    firrtl.connect %w, %addr : !firrtl.uint<4>, !firrtl.uint<4>
  }
  firrtl.connect %v, %ramport : !firrtl.uint<32>, !firrtl.uint<32>
}

// When the address is a node, the enable should be inferred where the address is declared.
firrtl.module @EnableInference1(in %p: !firrtl.uint<1>, in %addr: !firrtl.uint<4>, in %clock: !firrtl.clock, out %v: !firrtl.uint<32>) {
  %ram = firrtl.seqmem Undefined  : !firrtl.cmemory<uint<32>, 16>
  %invalid_ui32 = firrtl.invalidvalue : !firrtl.uint<32>
  firrtl.connect %v, %invalid_ui32 : !firrtl.uint<32>, !firrtl.uint<32>
  // CHECK: [[ADDR:%.*]] = firrtl.subfield %ram_ramport(0)
  // CHECK: [[EN:%.*]] = firrtl.subfield %ram_ramport(1)
  // CHECK: firrtl.when %p
  firrtl.when %p  {
   // CHECK-NEXT: firrtl.connect [[EN]], %c1_ui1
   // CHECK-NEXT: %n = firrtl.node %addr
   // CHECK-NEXT: firrtl.connect [[ADDR]], %n
   // CHECK-NEXT: firrtl.connect %2, %clock
   // CHECK-NEXT: firrtl.connect %v, %3
    %n = firrtl.node %addr : !firrtl.uint<4>
    %ramport, %ramport_port = firrtl.memoryport Read %ram : (!firrtl.cmemory<uint<32>, 16>) -> (!firrtl.uint<32>, !firrtl.cmemoryport)
    firrtl.memoryport.access %ramport_port[%n], %clock : !firrtl.cmemoryport, !firrtl.uint<4>, !firrtl.clock
    firrtl.connect %v, %ramport : !firrtl.uint<32>, !firrtl.uint<32>
  }
}
}

