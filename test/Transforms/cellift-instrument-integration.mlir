// RUN: circt-opt --cellift-instrument %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Integration test: full ALU with all main comb operations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @alu
// CHECK-SAME: (in %a : i32, in %a_t : i32, in %b : i32, in %b_t : i32, in %op : i3, in %op_t : i3, out result : i32, out result_t : i32)
hw.module @alu(in %a : i32, in %b : i32, in %op : i3, out result : i32) {
  %c0 = hw.constant 0 : i3
  %c1 = hw.constant 1 : i3
  %c2 = hw.constant 2 : i3
  %c3 = hw.constant 3 : i3
  %c4 = hw.constant 4 : i3
  %c5 = hw.constant 5 : i3

  %is_add = comb.icmp eq %op, %c0 : i3
  %is_sub = comb.icmp eq %op, %c1 : i3
  %is_and = comb.icmp eq %op, %c2 : i3
  %is_or  = comb.icmp eq %op, %c3 : i3
  %is_xor = comb.icmp eq %op, %c4 : i3
  %is_shl = comb.icmp eq %op, %c5 : i3

  %add_res = comb.add %a, %b : i32
  %sub_res = comb.sub %a, %b : i32
  %and_res = comb.and %a, %b : i32
  %or_res  = comb.or %a, %b : i32
  %xor_res = comb.xor %a, %b : i32
  %shl_res = comb.shl %a, %b : i32

  // Mux tree to select based on opcode.
  %m0 = comb.mux %is_shl, %shl_res, %xor_res : i32
  %m1 = comb.mux %is_xor, %xor_res, %m0 : i32
  %m2 = comb.mux %is_or, %or_res, %m1 : i32
  %m3 = comb.mux %is_and, %and_res, %m2 : i32
  %m4 = comb.mux %is_sub, %sub_res, %m3 : i32
  %result = comb.mux %is_add, %add_res, %m4 : i32

  // CHECK: hw.output {{%.+}}, {{%.+}} : i32, i32
  hw.output %result : i32
}

//===----------------------------------------------------------------------===//
// Integration test: register file (4 registers, 1 write port, 1 read port).
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @regfile
// CHECK-SAME: in %waddr : i2, in %waddr_t : i2
// CHECK-SAME: in %wdata : i8, in %wdata_t : i8
// CHECK-SAME: in %wen : i1, in %wen_t : i1
// CHECK-SAME: in %raddr : i2, in %raddr_t : i2
// CHECK-SAME: in %clk : !seq.clock
// CHECK-SAME: out rdata : i8, out rdata_t : i8
hw.module @regfile(in %waddr : i2, in %wdata : i8, in %wen : i1,
                   in %raddr : i2, in %clk : !seq.clock,
                   out rdata : i8) {
  // Decode write address.
  %c0 = hw.constant 0 : i2
  %c1 = hw.constant 1 : i2
  %c2 = hw.constant 2 : i2
  %c3 = hw.constant 3 : i2
  %w0 = comb.icmp eq %waddr, %c0 : i2
  %w1 = comb.icmp eq %waddr, %c1 : i2
  %w2 = comb.icmp eq %waddr, %c2 : i2
  %w3 = comb.icmp eq %waddr, %c3 : i2

  // Write enables.
  %we0 = comb.and %wen, %w0 : i1
  %we1 = comb.and %wen, %w1 : i1
  %we2 = comb.and %wen, %w2 : i1
  %we3 = comb.and %wen, %w3 : i1

  // Register next values: mux between write data and current value.
  %c0_8 = hw.constant 0 : i8
  %next0 = comb.mux %we0, %wdata, %r0 : i8
  %next1 = comb.mux %we1, %wdata, %r1 : i8
  %next2 = comb.mux %we2, %wdata, %r2 : i8
  %next3 = comb.mux %we3, %wdata, %r3 : i8

  // Registers.
  %r0 = seq.compreg %next0, %clk {name = "reg0"} : i8
  %r1 = seq.compreg %next1, %clk {name = "reg1"} : i8
  %r2 = seq.compreg %next2, %clk {name = "reg2"} : i8
  %r3 = seq.compreg %next3, %clk {name = "reg3"} : i8

  // Read mux.
  %r0_sel = comb.icmp eq %raddr, %c0 : i2
  %r1_sel = comb.icmp eq %raddr, %c1 : i2
  %r2_sel = comb.icmp eq %raddr, %c2 : i2

  %rd01 = comb.mux %r0_sel, %r0, %r1 : i8
  %rd23 = comb.mux %r2_sel, %r2, %r3 : i8
  %rd_hi = comb.extract %raddr from 1 : (i2) -> i1
  %rdata = comb.mux %rd_hi, %rd23, %rd01 : i8

  hw.output %rdata : i8
}

//===----------------------------------------------------------------------===//
// Integration test: simple counter with enable and reset.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @counter
// CHECK-SAME: in %clk : !seq.clock
// CHECK-SAME: in %rst : i1, in %rst_t : i1
// CHECK-SAME: in %en : i1, in %en_t : i1
// CHECK-SAME: out count : i8, out count_t : i8
hw.module @counter(in %clk : !seq.clock, in %rst : i1, in %en : i1,
                   out count : i8) {
  %c0 = hw.constant 0 : i8
  %c1 = hw.constant 1 : i8

  %inc = comb.add %cnt, %c1 : i8
  %next = comb.mux %en, %inc, %cnt : i8

  %cnt = seq.compreg %next, %clk reset %rst, %c0 {name = "cnt"} : i8

  // CHECK: hw.output {{%.+}}, {{%.+}} : i8, i8
  hw.output %cnt : i8
}

//===----------------------------------------------------------------------===//
// Integration test: hierarchical design with child and parent modules.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @adder
// CHECK-SAME: (in %x : i16, in %x_t : i16, in %y : i16, in %y_t : i16, out sum : i16, out sum_t : i16)
hw.module @adder(in %x : i16, in %y : i16, out sum : i16) {
  %s = comb.add %x, %y : i16
  hw.output %s : i16
}

// CHECK-LABEL: hw.module @top_with_adder
// CHECK-SAME: (in %a : i16, in %a_t : i16, in %b : i16, in %b_t : i16, in %c : i16, in %c_t : i16, out result : i16, out result_t : i16)
hw.module @top_with_adder(in %a : i16, in %b : i16, in %c : i16, out result : i16) {
  // Instance should be updated with taint ports.
  // CHECK: hw.instance "add1" @adder(x: %a: i16, x_t: %a_t: i16, y: %b: i16, y_t: %b_t: i16) -> (sum: i16, sum_t: i16)
  %ab = hw.instance "add1" @adder(x: %a: i16, y: %b: i16) -> (sum: i16)
  // CHECK: hw.instance "add2" @adder(x: {{%.+}}: i16, x_t: {{%.+}}: i16, y: %c: i16, y_t: %c_t: i16) -> (sum: i16, sum_t: i16)
  %result = hw.instance "add2" @adder(x: %ab: i16, y: %c: i16) -> (sum: i16)
  hw.output %result : i16
}

//===----------------------------------------------------------------------===//
// Integration test: bit manipulation with concat and extract.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @bit_manip
// CHECK-SAME: (in %a : i8, in %a_t : i8, in %b : i8, in %b_t : i8, out hi : i4, out hi_t : i4, out wide : i16, out wide_t : i16)
hw.module @bit_manip(in %a : i8, in %b : i8, out hi : i4, out wide : i16) {
  %hi = comb.extract %a from 4 : (i8) -> i4
  %wide = comb.concat %a, %b : i8, i8
  hw.output %hi, %wide : i4, i16
}

//===----------------------------------------------------------------------===//
// Integration test: comparison-based control flow.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @comparator
// CHECK-SAME: (in %a : i8, in %a_t : i8, in %b : i8, in %b_t : i8, out gt : i1, out gt_t : i1, out eq : i1, out eq_t : i1, out max : i8, out max_t : i8)
hw.module @comparator(in %a : i8, in %b : i8, out gt : i1, out eq : i1, out max : i8) {
  %gt = comb.icmp ugt %a, %b : i8
  %eq = comb.icmp eq %a, %b : i8
  %max = comb.mux %gt, %a, %b : i8
  hw.output %gt, %eq, %max : i1, i1, i8
}
