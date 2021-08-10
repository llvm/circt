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
  %port0 = firrtl.memoryport Infer %ram, %addr, %clock : (!firrtl.cmemory<vector<uint<1>, 2>, 2>, !firrtl.uint<1>, !firrtl.clock) -> !firrtl.vector<uint<1>, 2>
  // Subindexing a port should not count as a "use".
  %port1 = firrtl.memoryport Infer %ram, %addr, %clock : (!firrtl.cmemory<vector<uint<1>, 2>, 2>, !firrtl.uint<1>, !firrtl.clock) -> !firrtl.vector<uint<1>, 2>
  %0 = firrtl.subindex %port1[1] : !firrtl.vector<uint<1>, 2>
}
// CHECK:      firrtl.module @UnusedMemPort(in %clock: !firrtl.clock, in %addr: !firrtl.uint<1>) {
// CHECK-NEXT: }

firrtl.module @InferRead(in %clock: !firrtl.clock, in %addr: !firrtl.uint<1>, out %out : !firrtl.uint<1>, in %vec : !firrtl.vector<uint<1>, 2>) {
  // CHECK: %ram_port = firrtl.mem Undefined  {depth = 2 : i64, name = "ram", portNames = ["port"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
  // CHECK: [[ADDR:%.*]] = firrtl.subfield %ram_port(0)
  // CHECK: firrtl.connect [[ADDR]], %invalid_ui1
  // CHECK: [[EN:%.*]] = firrtl.subfield %ram_port(1)
  // CHECK: firrtl.connect [[EN]], %c0_ui1
  // CHECK: [[CLOCK:%.*]] = firrtl.subfield %ram_port(2)
  // CHECK: firrtl.connect [[CLOCK]], %invalid_clock
  // CHECK: [[DATA:%.*]] = firrtl.subfield %ram_port(3)
  %ram = firrtl.combmem : !firrtl.cmemory<uint<1>, 2>

  // CHECK: firrtl.connect [[ADDR]], %addr
  // CHECK: firrtl.connect [[EN]], %c1_ui1
  // CHECK: firrtl.connect [[CLOCK]], %clock
  %port = firrtl.memoryport Infer %ram, %addr, %clock : (!firrtl.cmemory<uint<1>, 2>, !firrtl.uint<1>, !firrtl.clock) -> !firrtl.uint<1>

  // CHECK: %node = firrtl.node [[DATA]]
  %node = firrtl.node %port : !firrtl.uint<1>

  // CHECK: firrtl.connect %out, [[DATA]]
  firrtl.connect %out, %port : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.partialconnect %out, [[DATA]]
  firrtl.partialconnect %out, %port : !firrtl.uint<1>, !firrtl.uint<1>

  // TODO: How do you get FileCheck to accept "[[[DATA]]]"?
  // CHECK: firrtl.subaccess %vec{{\[}}[[DATA]]{{\]}} : !firrtl.vector<uint<1>, 2>, !firrtl.uint<1>
  firrtl.subaccess %vec[%port] : !firrtl.vector<uint<1>, 2>, !firrtl.uint<1>
}

firrtl.module @InferWrite(in %clock: !firrtl.clock, in %addr: !firrtl.uint<1>, in %in : !firrtl.uint<1>) {
  // CHECK: %ram_port = firrtl.mem Undefined {depth = 2 : i64, name = "ram", portNames = ["port"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<1>, mask: uint<1>>
  // CHECK: [[ADDR:%.*]] = firrtl.subfield %ram_port(0)
  // CHECK: firrtl.connect [[ADDR]], %invalid_ui1
  // CHECK: [[EN:%.*]] = firrtl.subfield %ram_port(1)
  // CHECK: firrtl.connect [[EN]], %c0_ui1
  // CHECK: [[CLOCK:%.*]] = firrtl.subfield %ram_port(2)
  // CHECK: firrtl.connect [[CLOCK]], %invalid_clock
  // CHECK: [[DATA:%.*]] = firrtl.subfield %ram_port(3)
  // CHECK: firrtl.connect [[DATA]], %invalid_ui1
  // CHECK: [[MASK:%.*]] = firrtl.subfield %ram_port(4)
  // CHECK: firrtl.connect [[MASK]], %invalid_ui1
  %ram = firrtl.combmem : !firrtl.cmemory<uint<1>, 2>

  // CHECK: firrtl.connect [[ADDR]], %addr
  // CHECK: firrtl.connect [[EN]], %c1_ui1
  // CHECK: firrtl.connect [[CLOCK]], %clock
  // CHECK: firrtl.connect [[MASK]], %c0_ui1
  %port = firrtl.memoryport Infer %ram, %addr, %clock : (!firrtl.cmemory<uint<1>, 2>, !firrtl.uint<1>, !firrtl.clock) -> !firrtl.uint<1>

  // CHECK: firrtl.connect [[MASK]], %c1_ui1
  // CHECK: firrtl.connect [[DATA]], %in
  firrtl.connect %port, %in : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.connect [[MASK]], %c1_ui1
  // CHECK: firrtl.partialconnect [[DATA]], %in
  firrtl.partialconnect %port, %in : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @InferReadWrite(in %clock: !firrtl.clock, in %addr: !firrtl.uint<1>, in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
  // CHECK: %ram_port = firrtl.mem Undefined  {depth = 2 : i64, name = "ram", portNames = ["port"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<1>, wmode: uint<1>,  wdata: uint<1>, wmask: uint<1>>
  // CHECK: [[ADDR:%.*]] = firrtl.subfield %ram_port(0)
  // CHECK: firrtl.connect [[ADDR]], %invalid_ui1
  // CHECK: [[EN:%.*]] = firrtl.subfield %ram_port(1)
  // CHECK: firrtl.connect [[EN]], %c0_ui1
  // CHECK: [[CLOCK:%.*]] = firrtl.subfield %ram_port(2)
  // CHECK: firrtl.connect [[CLOCK]], %invalid_clock
  // CHECK: [[RDATA:%.*]] = firrtl.subfield %ram_port(3)
  // CHECK: [[WMODE:%.*]] = firrtl.subfield %ram_port(4)
  // CHECK: firrtl.connect [[WMODE]], %c0_ui1
  // CHECK: [[WDATA:%.*]] = firrtl.subfield %ram_port(5)
  // CHECK: firrtl.connect [[WDATA]], %invalid_ui1
  // CHECK: [[WMASK:%.*]] = firrtl.subfield %ram_port(6)
  // CHECK: firrtl.connect [[WMASK]], %invalid_ui1
  %ram = firrtl.combmem : !firrtl.cmemory<uint<1>, 2>

  // CHECK: firrtl.connect [[ADDR]], %addr : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.connect [[EN]], %c1_ui1
  // CHECK: firrtl.connect [[CLOCK]], %clock
  // CHECK: firrtl.connect [[WMASK]], %c0_ui1
  %port = firrtl.memoryport Read %ram, %addr, %clock : (!firrtl.cmemory<uint<1>, 2>, !firrtl.uint<1>, !firrtl.clock) -> !firrtl.uint<1>

  // CHECK: firrtl.connect [[WMASK]], %c1_ui1
  // CHECK: firrtl.connect [[WMODE]], %c1_ui1
  // CHECK: firrtl.connect [[WDATA]], %in
  firrtl.connect %port, %in : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.connect %out, [[RDATA]] 
  firrtl.connect %out, %port : !firrtl.uint<1>, !firrtl.uint<1>
}

// Check that partial connect properly sets the write mask for the elements which are actually connected.
firrtl.module @PartialConnectWriteMask(in %clock: !firrtl.clock, in %addr: !firrtl.uint<1>, in %data : !firrtl.bundle<c: vector<uint<3>, 2>, a: uint<1>>) {
  // CHECK: %ram_port = firrtl.mem Undefined  {depth = 2 : i64, name = "ram", portNames = ["port"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<1>, b: uint<2>, c: vector<uint<3>, 3>>, mask: bundle<a: uint<1>, b: uint<1>, c: vector<uint<1>, 3>>>
  // CHECK: [[DATA:%.*]] = firrtl.subfield %ram_port(3)
  // CHECK: [[MASK:%.*]] = firrtl.subfield %ram_port(4)
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
  %ram = firrtl.combmem : !firrtl.cmemory<bundle<a: uint<1>, b: uint<2>, c: vector<uint<3>, 3>>, 2>

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
  %port = firrtl.memoryport Infer %ram, %addr, %clock : (!firrtl.cmemory<bundle<a: uint<1>, b: uint<2>, c: vector<uint<3>, 3>>, 2>, !firrtl.uint<1>, !firrtl.clock) -> !firrtl.bundle<a: uint<1>, b: uint<2>, c: vector<uint<3>, 3>>

  // CHECK: [[A:%.*]] = firrtl.subfield [[MASK]](0) : (!firrtl.bundle<a: uint<1>, b: uint<1>, c: vector<uint<1>, 3>>) -> !firrtl.uint<1>
  // CHECK: firrtl.connect [[A]], %c1_ui1 
  // CHECK: [[C:%.*]] = firrtl.subfield [[MASK]](2) : (!firrtl.bundle<a: uint<1>, b: uint<1>, c: vector<uint<1>, 3>>) -> !firrtl.vector<uint<1>, 3>
  // CHECK: [[C_0:%.*]] = firrtl.subindex [[C]][0] : !firrtl.vector<uint<1>, 3>
  // CHECK: firrtl.connect [[C_0]], %c1_ui1 
  // CHECK: [[C_1:%.*]] = firrtl.subindex [[C]][1] : !firrtl.vector<uint<1>, 3>
  // CHECK: firrtl.connect [[C_1]], %c1_ui1 
  // CHECK: firrtl.partialconnect [[DATA]], %data
  firrtl.partialconnect %port, %data : !firrtl.bundle<a: uint<1>, b: uint<2>, c: vector<uint<3>, 3>>, !firrtl.bundle<c: vector<uint<3>, 2>, a: uint<1>>
}

firrtl.module @WriteToSubfield(in %clock: !firrtl.clock, in %addr: !firrtl.uint<1>, in %value: !firrtl.uint<1>) {
  %ram = firrtl.combmem : !firrtl.cmemory<bundle<a: uint<1>, b: uint<1>>, 2>
  %port = firrtl.memoryport Infer %ram, %addr, %clock : (!firrtl.cmemory<bundle<a: uint<1>, b:uint<1>>, 2>, !firrtl.uint<1>, !firrtl.clock) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
  %port_b = firrtl.subfield %port(1) : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<1>
  // Check that only the subfield of the mask is written to.
  // CHECK: [[DATA:%.*]] = firrtl.subfield %ram_port(3) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<1>, b: uint<1>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
  // CHECK: [[MASK:%.*]] = firrtl.subfield %ram_port(4) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<1>, b: uint<1>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
  // CHECK: [[DATA_B:%.*]] = firrtl.subfield [[DATA]](1) : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<1>
  // CHECK: [[MASK_B:%.*]] = firrtl.subfield [[MASK]](1) : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<1>
  // CHECK: firrtl.connect [[MASK_B]], %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.connect [[DATA_B]], %value : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %port_b, %value : !firrtl.uint<1>, !firrtl.uint<1>
}

// Read and write from different subfields of the memory.  The memory as a
// whole should be inferred to read+write.
firrtl.module @ReadAndWriteToSubfield(in %clock: !firrtl.clock, in %addr: !firrtl.uint<1>, in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
  %ram = firrtl.combmem : !firrtl.cmemory<bundle<a: uint<1>, b: uint<1>>, 2>
  %port = firrtl.memoryport Infer %ram, %addr, %clock : (!firrtl.cmemory<bundle<a: uint<1>, b:uint<1>>, 2>, !firrtl.uint<1>, !firrtl.clock) -> !firrtl.bundle<a: uint<1>, b: uint<1>>

  // CHECK: [[RDATA:%.*]] = firrtl.subfield %ram_port(3) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<1>, b: uint<1>>, wmode: uint<1>, wdata: bundle<a: uint<1>, b: uint<1>>, wmask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
  // CHECK: [[WMODE:%.*]] = firrtl.subfield %ram_port(4) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<1>, b: uint<1>>, wmode: uint<1>, wdata: bundle<a: uint<1>, b: uint<1>>, wmask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<1>
  // CHECK: [[WDATA:%.*]] = firrtl.subfield %ram_port(5) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<1>, b: uint<1>>, wmode: uint<1>, wdata: bundle<a: uint<1>, b: uint<1>>, wmask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
  // CHECK: [[WMASK:%.*]] = firrtl.subfield %ram_port(6) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<1>, b: uint<1>>, wmode: uint<1>, wdata: bundle<a: uint<1>, b: uint<1>>, wmask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
  // CHECK: [[WDATA_A:%.*]] = firrtl.subfield [[WDATA]](0) : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<1>
  // CHECK: [[WMASK_A:%.*]] = firrtl.subfield [[WMASK]](0) : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<1>
  // CHECK: firrtl.connect [[WMASK_A]], %c1_ui1
  // CHECK: firrtl.connect [[WMODE]], %c1_ui1
  // CHECK: firrtl.connect [[WDATA_A]], %in
  %port_a = firrtl.subfield %port(0) : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<1>
  firrtl.connect %port_a, %in : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: [[RDATA_B:%.*]] = firrtl.subfield [[RDATA]](1) : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<1>
  // CHECK: firrtl.connect %out, [[RDATA_B]] : !firrtl.uint<1>, !firrtl.uint<1>
  %port_b = firrtl.subfield %port(1) : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<1>
  firrtl.connect %out, %port_b : !firrtl.uint<1>, !firrtl.uint<1>
}

// Check that ports are sorted in alphabetical order.
firrtl.module @SortedPorts(in %clock: !firrtl.clock, in %addr : !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
  // CHECK: portNames = ["a", "b", "c"]
  %ram = firrtl.combmem : !firrtl.cmemory<vector<uint<1>, 2>, 2>
  %c = firrtl.memoryport Read %ram, %addr, %clock : (!firrtl.cmemory<vector<uint<1>, 2>, 2>, !firrtl.uint<1>, !firrtl.clock) -> !firrtl.vector<uint<1>, 2>
  %a = firrtl.memoryport Write %ram, %addr, %clock : (!firrtl.cmemory<vector<uint<1>, 2>, 2>, !firrtl.uint<1>, !firrtl.clock) -> !firrtl.vector<uint<1>, 2>
  %b = firrtl.memoryport ReadWrite %ram, %addr, %clock  : (!firrtl.cmemory<vector<uint<1>, 2>, 2>, !firrtl.uint<1>, !firrtl.clock) -> !firrtl.vector<uint<1>, 2>
}

// Check that annotations are preserved.
firrtl.module @Annotations(in %clock: !firrtl.clock, in %addr : !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
  // CHECK: firrtl.mem Undefined 
  // CHECK-SAME: annotations = [{a = "a"}]
  // CHECK-SAME: portAnnotations = [
  // CHECK-SAME:   [{b = "b"}],
  // CHECK-SAME:   [{c = "c"}]
  // CHECK-SAME: ]
  // CHECK-SAME: portNames = ["port0", "port1"]
  %ram = firrtl.combmem {annotations = [{a = "a"}]} : !firrtl.cmemory<vector<uint<1>, 2>, 2>
  %port0 = firrtl.memoryport Read %ram, %addr, %clock {annotations = [{b = "b"}]} : (!firrtl.cmemory<vector<uint<1>, 2>, 2>, !firrtl.uint<1>, !firrtl.clock) -> !firrtl.vector<uint<1>, 2>
  %port1 = firrtl.memoryport Read %ram, %addr, %clock {annotations = [{c = "c"}]} : (!firrtl.cmemory<vector<uint<1>, 2>, 2>, !firrtl.uint<1>, !firrtl.clock) -> !firrtl.vector<uint<1>, 2>
}

// When the address is a wire, the enable should be inferred where the address
// is driven.
firrtl.module @EnableInference0(in %p: !firrtl.uint<1>, in %addr: !firrtl.uint<4>, in %clock: !firrtl.clock, out %v: !firrtl.uint<32>) {
  %w = firrtl.wire  : !firrtl.uint<4>
  %invalid_ui4 = firrtl.invalidvalue : !firrtl.uint<4>
  // This connect should not count as "driving" a value.  If it accidentally
  // inserts an enable here, we will get a use-before-def error, so it is
  // enough of a check that this compiles.
  firrtl.connect %w, %invalid_ui4 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: [[ADDR:%.*]] = firrtl.subfield %ram_ramport(0)
  // CHECK: [[EN:%.*]] = firrtl.subfield %ram_ramport(1)
  %ram = firrtl.seqmem Undefined  : !firrtl.cmemory<uint<32>, 16>
  %ramport = firrtl.memoryport Read %ram, %w, %clock  : (!firrtl.cmemory<uint<32>, 16>, !firrtl.uint<4>, !firrtl.clock) -> !firrtl.uint<32>

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
    %ramport = firrtl.memoryport Read %ram, %n, %clock  : (!firrtl.cmemory<uint<32>, 16>, !firrtl.uint<4>, !firrtl.clock) -> !firrtl.uint<32>
    firrtl.connect %v, %ramport : !firrtl.uint<32>, !firrtl.uint<32>
  }
}
}

