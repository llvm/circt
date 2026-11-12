// RUN: circt-opt %s -export-verilog -verify-diagnostics -o /dev/null | FileCheck %s --strict-whitespace

// Spilled expressions are named after their operation and their operands.
// CHECK-LABEL: module Operations(
hw.module @Operations(in %a: i4, in %b: i4, in %c: i4, in %s: i1, out o1: i8, out o2: i8) {
  %c0_i4 = hw.constant 0 : i4
  %c-1_i4 = hw.constant -1 : i4
  // CHECK: wire        _eq_a_b = b == a;
  %eq = comb.icmp eq %b, %a : i4
  // CHECK-NEXT: wire        _ne_a_b = b != a;
  %ne = comb.icmp ne %b, %a : i4
  // CHECK-NEXT: wire        _lt_b_a = b < a;
  %lt = comb.icmp ult %b, %a : i4
  // CHECK-NEXT: wire        _ge_b_a = $signed(b) >= $signed(a);
  %ge = comb.icmp sge %b, %a : i4
  // CHECK-NEXT: wire [3:0]  _and_a_b = b & a;
  %and = comb.and %b, %a : i4
  // CHECK-NEXT: wire [3:0]  _or_a_b_c = c | a | b;
  %or = comb.or %c, %a, %b : i4
  // CHECK-NEXT: wire [3:0]  _xor_a_b = b ^ a;
  %xor = comb.xor %b, %a : i4
  // CHECK-NEXT: wire [3:0]  _not_a = ~a;
  %not = comb.xor %a, %c-1_i4 : i4
  // CHECK-NEXT: wire [3:0]  _add_a_b = b + a;
  %add = comb.add %b, %a : i4
  // CHECK-NEXT: wire [3:0]  _sub_b_a = b - a;
  %sub = comb.sub %b, %a : i4
  // CHECK-NEXT: wire [3:0]  _mux_s = s ? a : b;
  %mux = comb.mux %s, %a, %b : i4
  // CHECK-NEXT: wire [3:0]  _rep_s = {4{s}};
  %rep = comb.replicate %s : (i1) -> i4
  // CHECK-NEXT: wire [7:0]  _zext_a = {4'h0, a};
  %zext = comb.concat %c0_i4, %a : i4, i4
  // CHECK-NEXT: wire [7:0]  _sext_a = {{[{][{]}}4{a[3]}}, a};
  %sign = comb.extract %a from 3 : (i4) -> i1
  %signs = comb.replicate %sign : (i1) -> i4
  %sext = comb.concat %signs, %a : i4, i4
  // CHECK-NEXT: wire [11:0] _cat_c_b = {c, a, b};
  %cat = comb.concat %c, %a, %b : i4, i4, i4
  %bits = comb.concat %eq, %ne, %lt, %ge, %eq, %ne, %lt, %ge : i1, i1, i1, i1, i1, i1, i1, i1
  %nibbles = comb.concat %and, %or, %xor, %not, %add, %sub, %mux, %rep, %and, %or, %xor, %not, %add, %sub, %mux, %rep : i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4
  %nibblesLow = comb.extract %nibbles from 0 : (i64) -> i8
  %catLow = comb.extract %cat from 2 : (i12) -> i8
  // CHECK-NEXT: wire [63:0] _cat_and_a_b_rep_s =
  // CHECK: wire [7:0]  _add_sext_a_zext_a = _zext_a + _sext_a;
  %sum = comb.add %zext, %sext : i8
  %sum2 = comb.add %sum, %zext, %sext : i8
  %r = comb.xor %bits, %nibblesLow, %catLow, %sum2 : i8
  hw.output %r, %sum : i8, i8
}

// An array element is named after the array and the index.
// CHECK-LABEL: module ArrayGet(
hw.module @ArrayGet(in %v: !hw.array<4xi1>, in %i: i2, in %d: i1) {
  %c1_i2 = hw.constant 1 : i2
  %r = sv.reg : !hw.inout<i1>
  // CHECK: wire _v_at_i = v[i];
  // CHECK-NEXT: always @(posedge _v_at_i)
  %get = hw.array_get %v[%i] : !hw.array<4xi1>, i2
  sv.always posedge %get {
    sv.passign %r, %d : i1
  }
  // CHECK: wire _v_at_1 = v[2'h1];
  // CHECK-NEXT: always @(posedge _v_at_1)
  %get1 = hw.array_get %v[%c1_i2] : !hw.array<4xi1>, i2
  sv.always posedge %get1 {
    sv.passign %r, %d : i1
  }
}

// A user name, a wire and a name hint keep their names; an expression with an
// unnamed operand keeps the generated name.
// CHECK-LABEL: module UserNamesWin(
hw.module @UserNamesWin(in %a: i4, in %b: i4, out o1: i4, out o2: i4, out o3: i4) {
  // CHECK: wire [3:0] hint = a & b;
  %hint = comb.and %a, %b {sv.namehint = "hint"} : i4
  // CHECK-NEXT: wire [3:0] w;
  // CHECK-NEXT: assign w = a | b;
  %w = sv.wire : !hw.inout<i4>
  %or = comb.or %a, %b : i4
  sv.assign %w, %or : i4
  %wRead = sv.read_inout %w : !hw.inout<i4>
  // CHECK-NEXT: wire [3:0] _GEN = {a[1:0], b[1:0]} ^ a;
  %aLow = comb.extract %a from 0 : (i4) -> i2
  %bLow = comb.extract %b from 0 : (i4) -> i2
  %cat = comb.concat %aLow, %bLow : i2, i2
  %xor = comb.xor %cat, %a : i4
  %0 = comb.add %hint, %hint : i4
  %1 = comb.add %xor, %xor : i4
  hw.output %0, %wRead, %1 : i4, i4, i4
}

// Names are uniqued with a numeric suffix, and a long name is truncated.
// CHECK-LABEL: module Collisions(
hw.module @Collisions(in %a: i4, in %b: i4, in %a_very_long_port_name: i4, in %another_long_port_name: i4, out o1: i4, out o2: i4) {
  // CHECK: wire [3:0] _and_a_b = a & b;
  // CHECK-NEXT: wire [3:0] _and_a_b_0 = a & b;
  %0 = comb.and %a, %b : i4
  %1 = comb.and %a, %b : i4
  // CHECK-NEXT: wire [3:0] _and_a_very_long_port_name_another_long =
  %2 = comb.and %a_very_long_port_name, %another_long_port_name : i4
  %3 = comb.add %0, %0, %2 : i4
  %4 = comb.add %1, %1, %2 : i4
  hw.output %3, %4 : i4, i4
}

// Inputs are named by their port also when outputs come between them.
// CHECK-LABEL: module InterleavedPorts(
hw.module @InterleavedPorts(in %a: i4, out o: i4, in %b: i4) {
  // CHECK: wire [3:0] _and_a_b = a & b;
  %0 = comb.and %a, %b : i4
  %1 = comb.add %0, %0 : i4
  hw.output %1 : i4
}

// A new expression does not rename the temporaries that were there before.
// CHECK-LABEL: module Before(
// CHECK:      wire [3:0] _and_a_b = a & b;
// CHECK-NEXT: wire [3:0] _or_a_b = a | b;
hw.module @Before(in %a: i4, in %b: i4, out o1: i4, out o2: i4) {
  %0 = comb.and %a, %b : i4
  %1 = comb.or %a, %b : i4
  %2 = comb.add %0, %0 : i4
  %3 = comb.add %1, %1 : i4
  hw.output %2, %3 : i4, i4
}

// CHECK-LABEL: module After(
// CHECK:      wire [3:0] _xor_a_b = a ^ b;
// CHECK-NEXT: wire [3:0] _and_a_b = a & b;
// CHECK-NEXT: wire [3:0] _or_a_b = a | b;
hw.module @After(in %a: i4, in %b: i4, out o1: i4, out o2: i4, out o3: i4) {
  %x = comb.xor %a, %b : i4
  %0 = comb.and %a, %b : i4
  %1 = comb.or %a, %b : i4
  %2 = comb.add %0, %0 : i4
  %3 = comb.add %1, %1 : i4
  %4 = comb.add %x, %x : i4
  hw.output %2, %3, %4 : i4, i4, i4
}
