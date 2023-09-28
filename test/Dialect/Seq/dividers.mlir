// RUN: circt-opt --lower-seq-to-sv %s | FileCheck %s
// RUN: circt-opt --lower-seq-to-sv --export-verilog %s -o /dev/null | FileCheck %s --check-prefix=VERILOG

// VERILOG-LABEL: module divide_by_0
// CHECK-LABEL: @divide_by_0
hw.module @divide_by_0(in %clock: !seq.clock, out by_2: !seq.clock) {

  // CHECK: hw.output %clock : i1
  %by_2 = seq.clock_div %clock by 0
  hw.output %by_2 : !seq.clock
}


// VERILOG-LABEL: module divide_by_2
// CHECK-LABEL: @divide_by_2
hw.module @divide_by_2(in %clock: !seq.clock, out by_2: !seq.clock) {

  // CHECK: [[REGISTER:%.+]] = sv.reg
  // CHECK: sv.always posedge %clock {
  // CHECK:   [[READ_ALWAYS:%.+]] = sv.read_inout [[REGISTER]] : !hw.inout<i1>
  // CHECK:   [[INVERTED:%.+]] = comb.xor [[READ_ALWAYS]], %true : i1
  // CHECK:   sv.bpassign [[REGISTER]], [[INVERTED]] : i1
  // CHECK: }
  // CHECK: [[READ_OUTPUT:%.+]] = sv.read_inout [[REGISTER]] : !hw.inout<i1>
  // CHECK: sv.initial {
  // CHECK:   sv.bpassign [[REGISTER]], %false : i1
  // CHECK: }
  // CHECK: hw.output [[READ_OUTPUT]] : i1

  // VERILOG: reg clock_out_0;
  // VERILOG: always @(posedge clock)
  // VERILOG:   clock_out_0 = ~clock_out_0;
  // VERILOG: assign by_2 = clock_out_0;

  %by_2 = seq.clock_div %clock by 1
  hw.output %by_2 : !seq.clock
}

// VERILOG-LABEL: module divide_by_8
// CHECK-LABEL: @divide_by_8
hw.module @divide_by_8(in %clock: !seq.clock, out by_8: !seq.clock) {
  // CHECK: [[REGISTER_0:%.+]] = sv.reg : !hw.inout<i1>
  // CHECK: sv.always posedge %clock {
  // CHECK:   [[REGISTER_0_READ:%.+]] = sv.read_inout [[REGISTER_0]] : !hw.inout<i1>
  // CHECK:   [[INVERTED_0:%.+]] = comb.xor [[REGISTER_0_READ]], %true : i1
  // CHECK:   sv.bpassign [[REGISTER_0]], [[INVERTED_0]] : i1
  // CHECK: }
  // CHECK: [[REGISTER_0_OUT:%.+]] = sv.read_inout [[REGISTER_0]] : !hw.inout<i1>
  // CHECK: [[REGISTER_1:%.+]] = sv.reg : !hw.inout<i1>
  // CHECK: sv.always posedge [[REGISTER_0_OUT]] {
  // CHECK:   [[REGISTER_1_READ:%.+]] = sv.read_inout [[REGISTER_1]] : !hw.inout<i1>
  // CHECK:   [[INVERTED_1:%.+]] = comb.xor [[REGISTER_1_READ]], %true : i1
  // CHECK:   sv.bpassign [[REGISTER_1]], [[INVERTED_1]] : i1
  // CHECK: }
  // CHECK: [[REGISTER_1_OUT:%.+]] = sv.read_inout [[REGISTER_1]] : !hw.inout<i1>
  // CHECK: [[REGISTER_2:%.+]] = sv.reg : !hw.inout<i1>
  // CHECK: sv.always posedge [[REGISTER_1_OUT]] {
  // CHECK:   [[REGISTER_2_READ:%.+]] = sv.read_inout [[REGISTER_2]] : !hw.inout<i1>
  // CHECK:   [[INVERTED_2:%.+]] = comb.xor [[REGISTER_2_READ]], %true : i1
  // CHECK:   sv.bpassign [[REGISTER_2]], [[INVERTED_2]] : i1
  // CHECK: }
  // CHECK: [[REGISTER_2_OUT:%.+]] = sv.read_inout [[REGISTER_2]] : !hw.inout<i1>
  // CHECK: sv.initial {
  // CHECK:   sv.bpassign [[REGISTER_0]], %false : i1
  // CHECK:   sv.bpassign [[REGISTER_1]], %false : i1
  // CHECK:   sv.bpassign [[REGISTER_2]], %false : i1
  // CHECK: }
  // CHECK: hw.output [[REGISTER_2_OUT]] : i1

  // VERILOG: reg clock_out_0;
  // VERILOG: always @(posedge clock)
  // VERILOG:   clock_out_0 = ~clock_out_0;
  // VERILOG: reg clock_out_1;
  // VERILOG: always @(posedge clock_out_0)
  // VERILOG:   clock_out_1 = ~clock_out_1;
  // VERILOG: reg clock_out_2;
  // VERILOG: always @(posedge clock_out_1)
  // VERILOG:   clock_out_2 = ~clock_out_2;
  // VERILOG: initial begin
  // VERILOG:   clock_out_0 = 1'h0;
  // VERILOG:   clock_out_1 = 1'h0;
  // VERILOG:   clock_out_2 = 1'h0;
  // VERILOG: end
  // VERILOG: assign by_8 = clock_out_2;

  %by_8 = seq.clock_div %clock by 3
  hw.output %by_8 : !seq.clock
}

