// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types))' --allow-unregistered-dialect %s | FileCheck --check-prefixes=CHECK,LT,COMMON %s
// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types{preserve-aggregate=all}))' --allow-unregistered-dialect %s | FileCheck --check-prefixes=AGGREGATE,LT,COMMON %s
// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-signatures))' --allow-unregistered-dialect %s | FileCheck --check-prefixes=SIG,COMMON %s

firrtl.circuit "TopLevel" {

  // COMMON-LABEL: firrtl.module private @Simple
  // COMMON-SAME: in %[[SOURCE_VALID_NAME:source_valid]]: [[SOURCE_VALID_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: out %[[SOURCE_READY_NAME:source_ready]]: [[SOURCE_READY_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: in %[[SOURCE_DATA_NAME:source_data]]: [[SOURCE_DATA_TYPE:!firrtl.uint<64>]]
  // COMMON-SAME: out %[[SINK_VALID_NAME:sink_valid]]: [[SINK_VALID_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: in %[[SINK_READY_NAME:sink_ready]]: [[SINK_READY_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: out %[[SINK_DATA_NAME:sink_data]]: [[SINK_DATA_TYPE:!firrtl.uint<64>]]
  firrtl.module private @Simple(in %source: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
                        out %sink: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {

    // SIG-NEXT: %source = firrtl.wire interesting_name : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    // SIG-NEXT: %sink = firrtl.wire interesting_name : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    // LT-NEXT: firrtl.when %[[SOURCE_VALID_NAME]] : !firrtl.uint<1>
    // LT-NEXT:   firrtl.connect %[[SINK_DATA_NAME]], %[[SOURCE_DATA_NAME]] : [[SINK_DATA_TYPE]]
    // LT-NEXT:   firrtl.connect %[[SINK_VALID_NAME]], %[[SOURCE_VALID_NAME]] : [[SINK_VALID_TYPE]]
    // LT-NEXT:   firrtl.connect %[[SOURCE_READY_NAME]], %[[SINK_READY_NAME]] : [[SOURCE_READY_TYPE]]

    %0 = firrtl.subfield %source[valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    %1 = firrtl.subfield %source[ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    %2 = firrtl.subfield %source[data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    %3 = firrtl.subfield %sink[valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    %4 = firrtl.subfield %sink[ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    %5 = firrtl.subfield %sink[data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    firrtl.when %0 : !firrtl.uint<1> {
      firrtl.connect %5, %2 : !firrtl.uint<64>, !firrtl.uint<64>
      firrtl.connect %3, %0 : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %1, %4 : !firrtl.uint<1>, !firrtl.uint<1>
    }
  }

  // COMMON-LABEL: firrtl.module @TopLevel
  // COMMON-SAME: in %source_valid: [[SOURCE_VALID_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: out %source_ready: [[SOURCE_READY_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: in %source_data: [[SOURCE_DATA_TYPE:!firrtl.uint<64>]]
  // COMMON-SAME: out %sink_valid: [[SINK_VALID_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: in %sink_ready: [[SINK_READY_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: out %sink_data: [[SINK_DATA_TYPE:!firrtl.uint<64>]]
  firrtl.module @TopLevel(in %source: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
                          out %sink: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
    // SIG: %source = firrtl.wire interesting_name : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    // SIG: %sink = firrtl.wire interesting_name : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    // COMMON: %inst_source_valid, %inst_source_ready, %inst_source_data, %inst_sink_valid, %inst_sink_ready, %inst_sink_data
    // COMMON-SAME: = firrtl.instance "" @Simple(
    // COMMON-SAME: in source_valid: !firrtl.uint<1>, out source_ready: !firrtl.uint<1>, in source_data: !firrtl.uint<64>, out sink_valid: !firrtl.uint<1>, in sink_ready: !firrtl.uint<1>, out sink_data: !firrtl.uint<64>
    %sourceV, %sinkV = firrtl.instance "" @Simple(in source: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
                        out sink: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)

    // LT-NEXT: firrtl.matchingconnect %inst_source_valid, %source_valid
    // LT-NEXT: firrtl.matchingconnect %source_ready, %inst_source_ready
    // LT-NEXT: firrtl.matchingconnect %inst_source_data, %source_data
    // LT-NEXT: firrtl.matchingconnect %sink_valid, %inst_sink_valid
    // LT-NEXT: firrtl.matchingconnect %inst_sink_ready, %sink_ready
    // LT-NEXT: firrtl.matchingconnect %sink_data, %inst_sink_data
    firrtl.connect %sourceV, %source : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>

    firrtl.connect %sink, %sinkV : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
  }

  // COMMON-LABEL: firrtl.module private @Recursive
  // CHECK-SAME: in %[[FLAT_ARG_1_NAME:arg_foo_bar_baz]]: [[FLAT_ARG_1_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: in %[[FLAT_ARG_2_NAME:arg_foo_qux]]: [[FLAT_ARG_2_TYPE:!firrtl.sint<64>]]
  // CHECK-SAME: out %[[OUT_1_NAME:out1]]: [[OUT_1_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: out %[[OUT_2_NAME:out2]]: [[OUT_2_TYPE:!firrtl.sint<64>]]
  // AGGREGATE-SAME: in %[[ARG_NAME:arg]]: [[ARG_TYPE:!firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>]]
  // AGGREGATE-SAME: out %[[OUT_1_NAME:out1]]: [[OUT_1_TYPE:!firrtl.uint<1>]]
  // AGGREGATE-SAME: out %[[OUT_2_NAME:out2]]: [[OUT_2_TYPE:!firrtl.sint<64>]]
  // SIG-SAME: in %[[ARG_NAME:arg]]: [[ARG_TYPE:!firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>]]
  // SIG-SAME: out %[[OUT_1_NAME:out1]]: [[OUT_1_TYPE:!firrtl.uint<1>]]
  // SIG-SAME: out %[[OUT_2_NAME:out2]]: [[OUT_2_TYPE:!firrtl.sint<64>]]
  firrtl.module private @Recursive(in %arg: !firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>,
                           out %out1: !firrtl.uint<1>, out %out2: !firrtl.sint<64>) {

    // CHECK-NEXT: firrtl.connect %[[OUT_1_NAME]], %[[FLAT_ARG_1_NAME]] : [[OUT_1_TYPE]]
    // CHECK-NEXT: firrtl.connect %[[OUT_2_NAME]], %[[FLAT_ARG_2_NAME]] : [[OUT_2_TYPE]]
    // AGGREGATE-NEXT:  %0 = firrtl.subfield %[[ARG_NAME]][foo]
    // AGGREGATE-NEXT:  %1 = firrtl.subfield %0[bar]
    // AGGREGATE-NEXT:  %2 = firrtl.subfield %1[baz]
    // AGGREGATE-NEXT:  %3 = firrtl.subfield %0[qux]
    // AGGREGATE-NEXT:  firrtl.connect %[[OUT_1_NAME]], %2
    // AGGREGATE-NEXT:  firrtl.connect %[[OUT_2_NAME]], %3

    %0 = firrtl.subfield %arg[foo] : !firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>
    %1 = firrtl.subfield %0[bar] : !firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>
    %2 = firrtl.subfield %1[baz] : !firrtl.bundle<baz: uint<1>>
    %3 = firrtl.subfield %0[qux] : !firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>
    firrtl.connect %out1, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out2, %3 : !firrtl.sint<64>, !firrtl.sint<64>
  }

  // COMMON-LABEL: firrtl.module private @Uniquification
  // CHECK-SAME: in %[[FLATTENED_ARG:a_b]]: [[FLATTENED_TYPE:!firrtl.uint<1>]],
  // CHECK-NOT: %[[FLATTENED_ARG]]
  // CHECK-SAME: in %[[RENAMED_ARG:a_b.+]]: [[RENAMED_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: {portNames = ["a_b", "a_b"]}
  firrtl.module private @Uniquification(in %a: !firrtl.bundle<b: uint<1>>, in %a_b: !firrtl.uint<1>) {
  }

  // CHECK-LABEL: firrtl.module private @Top
  firrtl.module private @Top(in %in : !firrtl.bundle<a: uint<1>, b: uint<1>>,
                     out %out : !firrtl.bundle<a: uint<1>, b: uint<1>>) {
    // CHECK: firrtl.matchingconnect %out_a, %in_a : !firrtl.uint<1>
    // CHECK: firrtl.matchingconnect %out_b, %in_b : !firrtl.uint<1>
    // SIG: firrtl.connect %out, %in : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.connect %out, %in : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
  }

  // COMMON-LABEL: firrtl.module private @Foo
  // CHECK-SAME: in %[[FLAT_ARG_INPUT_NAME:a_b_c]]: [[FLAT_ARG_INPUT_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: out %[[FLAT_ARG_OUTPUT_NAME:b_b_c]]: [[FLAT_ARG_OUTPUT_TYPE:!firrtl.uint<1>]]
  firrtl.module private @Foo(in %a: !firrtl.bundle<b: bundle<c: uint<1>>>, out %b: !firrtl.bundle<b: bundle<c: uint<1>>>) {
    // CHECK: firrtl.matchingconnect %[[FLAT_ARG_OUTPUT_NAME]], %[[FLAT_ARG_INPUT_NAME]] : [[FLAT_ARG_OUTPUT_TYPE]]
    // SIG: firrtl.connect %b, %a : !firrtl.bundle<b: bundle<c: uint<1>>>
    firrtl.connect %b, %a : !firrtl.bundle<b: bundle<c: uint<1>>>, !firrtl.bundle<b: bundle<c: uint<1>>>
  }

// Test lower of a 1-read 1-write aggregate memory
//
// circuit Foo :
//   module Foo :
//     input clock: Clock
//     input rAddr: UInt<4>
//     input rEn: UInt<1>
//     output rData: {a: UInt<8>, b: UInt<8>}
//     input wAddr: UInt<4>
//     input wEn: UInt<1>
//     input wMask: {a: UInt<1>, b: UInt<1>}
//     input wData: {a: UInt<8>, b: UInt<8>}
//
//     mem memory:
//       data-type => {a: UInt<8>, b: UInt<8>}
//       depth => 16
//       reader => r
//       writer => w
//       read-latency => 0
//       write-latency => 1
//       read-under-write => undefined
//
//     memory.r.clk <= clock
//     memory.r.en <= rEn
//     memory.r.addr <= rAddr
//     rData <= memory.r.data
//
//     memory.w.clk <= clock
//     memory.w.en <= wEn
//     memory.w.addr <= wAddr
//     memory.w.mask <= wMask
//     memory.w.data <= wData

  // SIG-LABEL: firrtl.module private @Mem2
  firrtl.module private @Mem2(in %clock: !firrtl.clock, in %rAddr: !firrtl.uint<4>, in %rEn: !firrtl.uint<1>, out %rData: !firrtl.bundle<a: uint<8>, b: uint<8>>, in %wAddr: !firrtl.uint<4>, in %wEn: !firrtl.uint<1>, in %wMask: !firrtl.bundle<a: uint<1>, b: uint<1>>, in %wData: !firrtl.bundle<a: uint<8>, b: uint<8>>) {
    %memory_r, %memory_w = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r", "w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    %0 = firrtl.subfield %memory_r[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
    firrtl.connect %0, %clock : !firrtl.clock, !firrtl.clock
    %1 = firrtl.subfield %memory_r[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
    firrtl.connect %1, %rEn : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.subfield %memory_r[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
    firrtl.connect %2, %rAddr : !firrtl.uint<4>, !firrtl.uint<4>
    %3 = firrtl.subfield %memory_r[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
    firrtl.connect %rData, %3 : !firrtl.bundle<a: uint<8>, b: uint<8>>, !firrtl.bundle<a: uint<8>, b: uint<8>>
    %4 = firrtl.subfield %memory_w[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    firrtl.connect %4, %clock : !firrtl.clock, !firrtl.clock
    %5 = firrtl.subfield %memory_w[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    firrtl.connect %5, %wEn : !firrtl.uint<1>, !firrtl.uint<1>
    %6 = firrtl.subfield %memory_w[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    firrtl.connect %6, %wAddr : !firrtl.uint<4>, !firrtl.uint<4>
    %7 = firrtl.subfield %memory_w[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    firrtl.connect %7, %wMask : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
    %8 = firrtl.subfield %memory_w[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    firrtl.connect %8, %wData : !firrtl.bundle<a: uint<8>, b: uint<8>>, !firrtl.bundle<a: uint<8>, b: uint<8>>

    // ---------------------------------------------------------------------------------
    // Split memory "a" should exist
    // CHECK: %[[MEMORY_A_R:.+]], %[[MEMORY_A_W:.+]] = firrtl.mem {{.+}} data: uint<8>, mask: uint<1>
    //
    // Split memory "b" should exist
    // CHECK-NEXT: %[[MEMORY_B_R:.+]], %[[MEMORY_B_W:.+]] = firrtl.mem {{.+}} data: uint<8>, mask: uint<1>
    // ---------------------------------------------------------------------------------
    // Read ports
    // CHECK-NEXT: %[[MEMORY_A_R_ADDR:.+]] = firrtl.subfield %[[MEMORY_A_R]][addr]
    // CHECK-NEXT: firrtl.matchingconnect %[[MEMORY_A_R_ADDR]], %[[MEMORY_R_ADDR:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_R_ADDR:.+]] = firrtl.subfield %[[MEMORY_B_R]][addr]
    // CHECK-NEXT: firrtl.matchingconnect %[[MEMORY_B_R_ADDR]], %[[MEMORY_R_ADDR]]
    // CHECK-NEXT: %[[MEMORY_A_R_EN:.+]] = firrtl.subfield %[[MEMORY_A_R]][en]
    // CHECK-NEXT: firrtl.matchingconnect %[[MEMORY_A_R_EN]], %[[MEMORY_R_EN:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_R_EN:.+]] = firrtl.subfield %[[MEMORY_B_R]][en]
    // CHECK-NEXT: firrtl.matchingconnect %[[MEMORY_B_R_EN]], %[[MEMORY_R_EN]]
    // CHECK-NEXT: %[[MEMORY_A_R_CLK:.+]] = firrtl.subfield %[[MEMORY_A_R]][clk]
    // CHECK-NEXT: firrtl.matchingconnect %[[MEMORY_A_R_CLK]], %[[MEMORY_R_CLK:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_R_CLK:.+]] = firrtl.subfield %[[MEMORY_B_R]][clk]
    // CHECK-NEXT: firrtl.matchingconnect %[[MEMORY_B_R_CLK]], %[[MEMORY_R_CLK]]
    // CHECK-NEXT: %[[MEMORY_A_R_DATA:.+]] = firrtl.subfield %[[MEMORY_A_R]][data]
    // CHECK-NEXT: firrtl.matchingconnect %[[WIRE_A_R_DATA:.+]], %[[MEMORY_A_R_DATA]] :
    // CHECK-NEXT: %[[MEMORY_B_R_DATA:.+]] = firrtl.subfield %[[MEMORY_B_R]][data]
    // CHECK-NEXT: firrtl.matchingconnect %[[WIRE_B_R_DATA:.+]], %[[MEMORY_B_R_DATA]] :
    // ---------------------------------------------------------------------------------
    // Write Ports
    // CHECK-NEXT: %[[MEMORY_A_W_ADDR:.+]] = firrtl.subfield %[[MEMORY_A_W]][addr]
    // CHECK-NEXT: firrtl.matchingconnect %[[MEMORY_A_W_ADDR]], %[[MEMORY_W_ADDR:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_W_ADDR:.+]] = firrtl.subfield %[[MEMORY_B_W]][addr]
    // CHECK-NEXT: firrtl.matchingconnect %[[MEMORY_B_W_ADDR]], %[[MEMORY_W_ADDR]] :
    // CHECK-NEXT: %[[MEMORY_A_W_EN:.+]] = firrtl.subfield %[[MEMORY_A_W]][en]
    // CHECK-NEXT: firrtl.matchingconnect %[[MEMORY_A_W_EN]], %[[MEMORY_W_EN:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_W_EN:.+]] = firrtl.subfield %[[MEMORY_B_W]][en]
    // CHECK-NEXT: firrtl.matchingconnect %[[MEMORY_B_W_EN]], %[[MEMORY_W_EN]] :
    // CHECK-NEXT: %[[MEMORY_A_W_CLK:.+]] = firrtl.subfield %[[MEMORY_A_W]][clk]
    // CHECK-NEXT: firrtl.matchingconnect %[[MEMORY_A_W_CLK]], %[[MEMORY_W_CLK:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_W_CLK:.+]] = firrtl.subfield %[[MEMORY_B_W]][clk]
    // CHECK-NEXT: firrtl.matchingconnect %[[MEMORY_B_W_CLK]], %[[MEMORY_W_CLK]] :
    // CHECK-NEXT: %[[MEMORY_A_W_DATA:.+]] = firrtl.subfield %[[MEMORY_A_W]][data]
    // CHECK-NEXT: firrtl.matchingconnect %[[MEMORY_A_W_DATA]], %[[WIRE_A_W_DATA:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_W_DATA:.+]] = firrtl.subfield %[[MEMORY_B_W]][data]
    // CHECK-NEXT: firrtl.matchingconnect %[[MEMORY_B_W_DATA]], %[[WIRE_B_W_DATA:.+]] :
    // CHECK-NEXT: %[[MEMORY_A_W_MASK:.+]] = firrtl.subfield %[[MEMORY_A_W]][mask]
    // CHECK-NEXT: firrtl.matchingconnect %[[MEMORY_A_W_MASK]], %[[WIRE_A_W_MASK:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_W_MASK:.+]] = firrtl.subfield %[[MEMORY_B_W]][mask]
    // CHECK-NEXT: firrtl.matchingconnect %[[MEMORY_B_W_MASK]], %[[WIRE_B_W_MASK:.+]] :
    //
    // Connections to module ports
    // CHECK-NEXT: firrtl.connect %[[MEMORY_R_CLK]], %clock
    // CHECK-NEXT: firrtl.connect %[[MEMORY_R_EN]], %rEn
    // CHECK-NEXT: firrtl.connect %[[MEMORY_R_ADDR]], %rAddr
    // CHECK-NEXT: firrtl.matchingconnect %rData_a, %[[WIRE_A_R_DATA]]
    // CHECK-NEXT: firrtl.matchingconnect %rData_b, %[[WIRE_B_R_DATA]]
    // CHECK-NEXT: firrtl.connect %[[MEMORY_W_CLK]], %clock
    // CHECK-NEXT: firrtl.connect %[[MEMORY_W_EN]], %wEn
    // CHECK-NEXT: firrtl.connect %[[MEMORY_W_ADDR]], %wAddr
    // CHECK-NEXT: firrtl.matchingconnect %[[WIRE_A_W_MASK]], %wMask_a
    // CHECK-NEXT: firrtl.matchingconnect %[[WIRE_B_W_MASK]], %wMask_b
    // CHECK-NEXT: firrtl.matchingconnect %[[WIRE_A_W_DATA]], %wData_a
    // CHECK-NEXT: firrtl.matchingconnect %[[WIRE_B_W_DATA]], %wData_b
  }


// https://github.com/llvm/circt/issues/593

    firrtl.module private @mod_2(in %clock: !firrtl.clock, in %inp_a: !firrtl.bundle<inp_d: uint<14>>) {
    }
    firrtl.module private @top_mod(in %clock: !firrtl.clock) {
      %U0_clock, %U0_inp_a = firrtl.instance U0 @mod_2(in clock: !firrtl.clock, in inp_a: !firrtl.bundle<inp_d: uint<14>>)
      %0 = firrtl.invalidvalue : !firrtl.clock
      firrtl.connect %U0_clock, %0 : !firrtl.clock, !firrtl.clock
      %1 = firrtl.invalidvalue : !firrtl.bundle<inp_d: uint<14>>
      firrtl.connect %U0_inp_a, %1 : !firrtl.bundle<inp_d: uint<14>>, !firrtl.bundle<inp_d: uint<14>>
    }



//CHECK-LABEL:     firrtl.module private @mod_2(in %clock: !firrtl.clock, in %inp_a_inp_d: !firrtl.uint<14>)
//CHECK:    firrtl.module private @top_mod(in %clock: !firrtl.clock)
//CHECK-NEXT:      %U0_clock, %U0_inp_a_inp_d = firrtl.instance U0 @mod_2(in clock: !firrtl.clock, in inp_a_inp_d: !firrtl.uint<14>)
//CHECK-NEXT:      %invalid_clock = firrtl.invalidvalue : !firrtl.clock
//CHECK-NEXT:      firrtl.connect %U0_clock, %invalid_clock : !firrtl.clock
//CHECK-NEXT:      %invalid_ui14 = firrtl.invalidvalue : !firrtl.uint<14>
//CHECK-NEXT:      firrtl.matchingconnect %U0_inp_a_inp_d, %invalid_ui14 : !firrtl.uint<14>

//AGGREGATE-LABEL: firrtl.module private @mod_2(in %clock: !firrtl.clock, in %inp_a: !firrtl.bundle<inp_d: uint<14>>)
//AGGREGATE:    firrtl.module private @top_mod(in %clock: !firrtl.clock)
//AGGREGATE-NEXT:  %U0_clock, %U0_inp_a = firrtl.instance U0  @mod_2(in clock: !firrtl.clock, in inp_a: !firrtl.bundle<inp_d: uint<14>>)
//AGGREGATE-NEXT:  %invalid_clock = firrtl.invalidvalue : !firrtl.clock
//AGGREGATE-NEXT:  firrtl.connect %U0_clock, %invalid_clock : !firrtl.clock
//AGGREGATE-NEXT:  %invalid = firrtl.invalidvalue : !firrtl.bundle<inp_d: uint<14>>
//AGGREGATE-NEXT:  %0 = firrtl.subfield %invalid[inp_d] : !firrtl.bundle<inp_d: uint<14>>
//AGGREGATE-NEXT:  %1 = firrtl.subfield %U0_inp_a[inp_d] : !firrtl.bundle<inp_d: uint<14>>
//AGGREGATE-NEXT:  firrtl.matchingconnect %1, %0 : !firrtl.uint<14>
//SIG-LABEL: firrtl.module private @mod_2(in %clock: !firrtl.clock, in %inp_a: !firrtl.bundle<inp_d: uint<14>>)
//SIG:    firrtl.module private @top_mod(in %clock: !firrtl.clock)

// https://github.com/llvm/circt/issues/661

// This test is just checking that the following doesn't error.
    // COMMON-LABEL: firrtl.module private @Issue661
    firrtl.module private @Issue661(in %clock: !firrtl.clock) {
      %head_MPORT_2, %head_MPORT_6 = firrtl.mem Undefined {depth = 20 : i64, name = "head", portNames = ["MPORT_2", "MPORT_6"], readLatency = 0 : i32, writeLatency = 1 : i32}
      : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<5>, mask: uint<1>>,
        !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<5>, mask: uint<1>>
      %127 = firrtl.subfield %head_MPORT_6[clk] : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<5>, mask: uint<1>>
    }

// Check that a non-bundled mux ops are untouched.
    // COMMON-LABEL: firrtl.module private @Mux
    firrtl.module private @Mux(in %p: !firrtl.uint<1>, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
      // CHECK-NEXT: %0 = firrtl.mux(%p, %a, %b) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %c, %0 : !firrtl.uint<1>
      %0 = firrtl.mux(%p, %a, %b) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      firrtl.connect %c, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    }
    // COMMON-LABEL: firrtl.module private @MuxBundle
    firrtl.module private @MuxBundle(in %p: !firrtl.uint<1>, in %a: !firrtl.bundle<a: uint<1>>, in %b: !firrtl.bundle<a: uint<1>>, out %c: !firrtl.bundle<a: uint<1>>) {
      // CHECK-NEXT: %0 = firrtl.mux(%p, %a_a, %b_a) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      // CHECK-NEXT: firrtl.matchingconnect %c_a, %0 : !firrtl.uint<1>
      // SIG:        firrtl.mux(%p, %a, %b) : (!firrtl.uint<1>, !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>) -> !firrtl.bundle<a: uint<1>>
      %0 = firrtl.mux(%p, %a, %b) : (!firrtl.uint<1>, !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>) -> !firrtl.bundle<a: uint<1>>
      firrtl.connect %c, %0 : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    }

    // COMMON-LABEL: firrtl.module private @NodeBundle
    firrtl.module private @NodeBundle(in %a: !firrtl.bundle<a: uint<1>>, out %b: !firrtl.uint<1>) {
      // CHECK-NEXT: %n_a = firrtl.node %a_a  : !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %b, %n_a : !firrtl.uint<1>
      // SIG:        firrtl.node %a : !firrtl.bundle<a: uint<1>>
      %n = firrtl.node %a : !firrtl.bundle<a: uint<1>>
      %n_a = firrtl.subfield %n[a] : !firrtl.bundle<a: uint<1>>
      firrtl.connect %b, %n_a : !firrtl.uint<1>, !firrtl.uint<1>
    }

    // CHECK-LABEL: firrtl.module private @RegBundle(in %a_a: !firrtl.uint<1>, in %clk: !firrtl.clock, out %b_a: !firrtl.uint<1>)
    firrtl.module private @RegBundle(in %a: !firrtl.bundle<a: uint<1>>, in %clk: !firrtl.clock, out %b: !firrtl.bundle<a: uint<1>>) {
      // CHECK-NEXT: %x_a = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %x_a, %a_a : !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %b_a, %x_a : !firrtl.uint<1>
      // SIG: %x = firrtl.reg %clk : !firrtl.clock, !firrtl.bundle<a: uint<1>>
      %x = firrtl.reg %clk {name = "x"} : !firrtl.clock, !firrtl.bundle<a: uint<1>>
      %0 = firrtl.subfield %x[a] : !firrtl.bundle<a: uint<1>>
      %1 = firrtl.subfield %a[a] : !firrtl.bundle<a: uint<1>>
      firrtl.connect %0, %1 : !firrtl.uint<1>, !firrtl.uint<1>
      %2 = firrtl.subfield %b[a] : !firrtl.bundle<a: uint<1>>
      %3 = firrtl.subfield %x[a] : !firrtl.bundle<a: uint<1>>
      firrtl.connect %2, %3 : !firrtl.uint<1>, !firrtl.uint<1>
    }

    // CHECK-LABEL: firrtl.module private @RegBundleWithBulkConnect(in %a_a: !firrtl.uint<1>, in %clk: !firrtl.clock, out %b_a: !firrtl.uint<1>)
    firrtl.module private @RegBundleWithBulkConnect(in %a: !firrtl.bundle<a: uint<1>>, in %clk: !firrtl.clock, out %b: !firrtl.bundle<a: uint<1>>) {
      // CHECK-NEXT: %x_a = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<1>
      // CHECK-NEXT: firrtl.matchingconnect %x_a, %a_a : !firrtl.uint<1>
      // CHECK-NEXT: firrtl.matchingconnect %b_a, %x_a : !firrtl.uint<1>
      %x = firrtl.reg %clk {name = "x"} : !firrtl.clock, !firrtl.bundle<a: uint<1>>
      firrtl.connect %x, %a : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
      firrtl.connect %b, %x : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    }

    // CHECK-LABEL: firrtl.module private @WireBundle(in %a_a: !firrtl.uint<1>,  out %b_a: !firrtl.uint<1>)
    firrtl.module private @WireBundle(in %a: !firrtl.bundle<a: uint<1>>,  out %b: !firrtl.bundle<a: uint<1>>) {
      // CHECK-NEXT: %x_a = firrtl.wire  : !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %x_a, %a_a : !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %b_a, %x_a : !firrtl.uint<1>
      %x = firrtl.wire : !firrtl.bundle<a: uint<1>>
      %0 = firrtl.subfield %x[a] : !firrtl.bundle<a: uint<1>>
      %1 = firrtl.subfield %a[a] : !firrtl.bundle<a: uint<1>>
      firrtl.connect %0, %1 : !firrtl.uint<1>, !firrtl.uint<1>
      %2 = firrtl.subfield %b[a] : !firrtl.bundle<a: uint<1>>
      %3 = firrtl.subfield %x[a] : !firrtl.bundle<a: uint<1>>
      firrtl.connect %2, %3 : !firrtl.uint<1>, !firrtl.uint<1>
    }

  // COMMON-LABEL: firrtl.module private @WireBundlesWithBulkConnect
  firrtl.module private @WireBundlesWithBulkConnect(in %source: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
                             out %sink: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
    // CHECK: %w_valid = firrtl.wire  : !firrtl.uint<1>
    // CHECK: %w_ready = firrtl.wire  : !firrtl.uint<1>
    // CHECK: %w_data = firrtl.wire  : !firrtl.uint<64>
    %w = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    // CHECK: firrtl.matchingconnect %w_valid, %source_valid : !firrtl.uint<1>
    // CHECK: firrtl.matchingconnect %source_ready, %w_ready : !firrtl.uint<1>
    // CHECK: firrtl.matchingconnect %w_data, %source_data : !firrtl.uint<64>
    firrtl.connect %w, %source : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    // CHECK: firrtl.matchingconnect %sink_valid, %w_valid : !firrtl.uint<1>
    // CHECK: firrtl.matchingconnect %w_ready, %sink_ready : !firrtl.uint<1>
    // CHECK: firrtl.matchingconnect %sink_data, %w_data : !firrtl.uint<64>
    firrtl.connect %sink, %w : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
  }

// Test vector lowering
  firrtl.module private @LowerVectors(in %a: !firrtl.vector<uint<1>, 2>, out %b: !firrtl.vector<uint<1>, 2>) {
    firrtl.connect %b, %a: !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
  // CHECK-LABEL: firrtl.module private @LowerVectors(in %a_0: !firrtl.uint<1>, in %a_1: !firrtl.uint<1>, out %b_0: !firrtl.uint<1>, out %b_1: !firrtl.uint<1>)
  // CHECK: firrtl.matchingconnect %b_0, %a_0
  // CHECK: firrtl.matchingconnect %b_1, %a_1
  // AGGREGATE-LABEL: firrtl.module private @LowerVectors(in %a: !firrtl.vector<uint<1>, 2>, out %b: !firrtl.vector<uint<1>, 2>)
  // AGGREGATE-NEXT: %0 = firrtl.subindex %a[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT: %1 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT: firrtl.matchingconnect %1, %0 : !firrtl.uint<1>
  // AGGREGATE-NEXT: %2 = firrtl.subindex %a[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT: %3 = firrtl.subindex %b[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT: firrtl.matchingconnect %3, %2 : !firrtl.uint<1>

// Test vector of bundles lowering
  // COMMON-LABEL: firrtl.module private @LowerVectorsOfBundles(in %in_0_a: !firrtl.uint<1>, out %in_0_b: !firrtl.uint<1>, in %in_1_a: !firrtl.uint<1>, out %in_1_b: !firrtl.uint<1>, out %out_0_a: !firrtl.uint<1>, in %out_0_b: !firrtl.uint<1>, out %out_1_a: !firrtl.uint<1>, in %out_1_b: !firrtl.uint<1>)
  firrtl.module private @LowerVectorsOfBundles(in %in: !firrtl.vector<bundle<a : uint<1>, b  flip: uint<1>>, 2>,
                                       out %out: !firrtl.vector<bundle<a : uint<1>, b  flip: uint<1>>, 2>) {
    // LT:      firrtl.matchingconnect %out_0_a, %in_0_a : !firrtl.uint<1>
    // LT-NEXT: firrtl.matchingconnect %in_0_b, %out_0_b : !firrtl.uint<1>
    // LT-NEXT: firrtl.matchingconnect %out_1_a, %in_1_a : !firrtl.uint<1>
    // LT-NEXT: firrtl.matchingconnect %in_1_b, %out_1_b : !firrtl.uint<1>
    firrtl.connect %out, %in: !firrtl.vector<bundle<a : uint<1>, b flip: uint<1>>, 2>, !firrtl.vector<bundle<a : uint<1>, b flip: uint<1>>, 2>
  }

  // COMMON-LABEL: firrtl.extmodule @ExternalModule(in source_valid: !firrtl.uint<1>, out source_ready: !firrtl.uint<1>, in source_data: !firrtl.uint<64>)
  firrtl.extmodule @ExternalModule(in source: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
  firrtl.module private @Test() {
    // COMMON: %inst_source_valid, %inst_source_ready, %inst_source_data = firrtl.instance "" @ExternalModule(in source_valid: !firrtl.uint<1>, out source_ready: !firrtl.uint<1>, in source_data: !firrtl.uint<64>)
    %inst_source = firrtl.instance "" @ExternalModule(in source: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
  }

// Test RegResetOp lowering
  // COMMON-LABEL: firrtl.module private @LowerRegResetOp
  firrtl.module private @LowerRegResetOp(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a_d: !firrtl.vector<uint<1>, 2>, out %a_q: !firrtl.vector<uint<1>, 2>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %init = firrtl.wire  : !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subindex %init[0] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = firrtl.subindex %init[1] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %r = firrtl.regreset %clock, %reset, %init {name = "r"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    firrtl.connect %r, %a_d : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    firrtl.connect %a_q, %r : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %init_0 = firrtl.wire  : !firrtl.uint<1>
  // CHECK:   %init_1 = firrtl.wire  : !firrtl.uint<1>
  // CHECK:   firrtl.connect %init_0, %c0_ui1 : !firrtl.uint<1>
  // CHECK:   firrtl.connect %init_1, %c0_ui1 : !firrtl.uint<1>
  // CHECK:   %r_0 = firrtl.regreset %clock, %reset, %init_0 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   %r_1 = firrtl.regreset %clock, %reset, %init_1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.matchingconnect %r_0, %a_d_0 : !firrtl.uint<1>
  // CHECK:   firrtl.matchingconnect %r_1, %a_d_1 : !firrtl.uint<1>
  // CHECK:   firrtl.matchingconnect %a_q_0, %r_0 : !firrtl.uint<1>
  // CHECK:   firrtl.matchingconnect %a_q_1, %r_1 : !firrtl.uint<1>
  // AGGREGATE:       %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // AGGREGATE-NEXT:  %init = firrtl.wire  : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  %0 = firrtl.subindex %init[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>
  // AGGREGATE-NEXT:  %1 = firrtl.subindex %init[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  firrtl.connect %1, %c0_ui1 : !firrtl.uint<1>
  // AGGREGATE-NEXT:  %r = firrtl.regreset %clock, %reset, %init  : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  %2 = firrtl.subindex %a_d[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  %3 = firrtl.subindex %r[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  firrtl.matchingconnect %3, %2 : !firrtl.uint<1>
  // AGGREGATE-NEXT:  %4 = firrtl.subindex %a_d[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  %5 = firrtl.subindex %r[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  firrtl.matchingconnect %5, %4 : !firrtl.uint<1>
  // AGGREGATE-NEXT:  %6 = firrtl.subindex %r[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  %7 = firrtl.subindex %a_q[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  firrtl.matchingconnect %7, %6 : !firrtl.uint<1>
  // AGGREGATE-NEXT:  %8 = firrtl.subindex %r[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  %9 = firrtl.subindex %a_q[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  firrtl.matchingconnect %9, %8 : !firrtl.uint<1>

// Test RegResetOp lowering without name attribute
// https://github.com/llvm/circt/issues/795
  // CHECK-LABEL: firrtl.module private @LowerRegResetOpNoName
  firrtl.module private @LowerRegResetOpNoName(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a_d: !firrtl.vector<uint<1>, 2>, out %a_q: !firrtl.vector<uint<1>, 2>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %init = firrtl.wire  : !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subindex %init[0] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = firrtl.subindex %init[1] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %r = firrtl.regreset %clock, %reset, %init {name = ""} : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    firrtl.connect %r, %a_d : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    firrtl.connect %a_q, %r : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %init_0 = firrtl.wire  : !firrtl.uint<1>
  // CHECK:   %init_1 = firrtl.wire  : !firrtl.uint<1>
  // CHECK:   firrtl.connect %init_0, %c0_ui1 : !firrtl.uint<1>
  // CHECK:   firrtl.connect %init_1, %c0_ui1 : !firrtl.uint<1>
  // CHECK:   %0 = firrtl.regreset %clock, %reset, %init_0 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   %1 = firrtl.regreset %clock, %reset, %init_1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.matchingconnect %0, %a_d_0 : !firrtl.uint<1>
  // CHECK:   firrtl.matchingconnect %1, %a_d_1 : !firrtl.uint<1>
  // CHECK:   firrtl.matchingconnect %a_q_0, %0 : !firrtl.uint<1>
  // CHECK:   firrtl.matchingconnect %a_q_1, %1 : !firrtl.uint<1>

// Test RegOp lowering without name attribute
// https://github.com/llvm/circt/issues/795
  // CHECK-LABEL: firrtl.module private @lowerRegOpNoName
  firrtl.module private @lowerRegOpNoName(in %clock: !firrtl.clock, in %a_d: !firrtl.vector<uint<1>, 2>, out %a_q: !firrtl.vector<uint<1>, 2>) {
    %r = firrtl.reg %clock {name = ""} : !firrtl.clock, !firrtl.vector<uint<1>, 2>
      firrtl.connect %r, %a_d : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
      firrtl.connect %a_q, %r : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
 // CHECK:    %0 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
 // CHECK:    %1 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
 // CHECK:    firrtl.matchingconnect %0, %a_d_0 : !firrtl.uint<1>
 // CHECK:    firrtl.matchingconnect %1, %a_d_1 : !firrtl.uint<1>
 // CHECK:    firrtl.matchingconnect %a_q_0, %0 : !firrtl.uint<1>
 // CHECK:    firrtl.matchingconnect %a_q_1, %1 : !firrtl.uint<1>

// Test that InstanceOp Annotations are copied to the new instance.
  firrtl.module private @Bar(out %a: !firrtl.vector<uint<1>, 2>) {
    %0 = firrtl.invalidvalue : !firrtl.vector<uint<1>, 2>
    firrtl.connect %a, %0 : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
  firrtl.module private @AnnotationsInstanceOp() {
    %bar_a = firrtl.instance bar {annotations = [{a = "a"}]} @Bar(out a: !firrtl.vector<uint<1>, 2>)
  }
  // CHECK: firrtl.instance
  // CHECK-SAME: annotations = [{a = "a"}]

// Test that MemOp Annotations are copied to lowered MemOps.
  // COMMON-LABEL: firrtl.module private @AnnotationsMemOp
  firrtl.module private @AnnotationsMemOp() {
    %bar_r, %bar_w = firrtl.mem Undefined  {annotations = [{a = "a"}], depth = 16 : i64, name = "bar", portNames = ["r", "w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: vector<uint<8>, 2>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: vector<uint<8>, 2>, mask: vector<uint<1>, 2>>
  }
  // LT: firrtl.mem
  // LT-SAME: annotations = [{a = "a"}]
  // LT: firrtl.mem
  // LT-SAME: annotations = [{a = "a"}]

// Test that WireOp Annotations are copied to lowered WireOps.
  // COMMON-LABEL: firrtl.module private @AnnotationsWireOp
  firrtl.module private @AnnotationsWireOp() {
    %bar = firrtl.wire  {annotations = [{a = "a"}]} : !firrtl.vector<uint<1>, 2>
  }
  // CHECK: firrtl.wire
  // CHECK-SAME: annotations = [{a = "a"}]
  // CHECK: firrtl.wire
  // CHECK-SAME: annotations = [{a = "a"}]

// Test that Reg/RegResetOp Annotations are copied to lowered registers.
  // CHECK-LABEL: firrtl.module private @AnnotationsRegOp
  firrtl.module private @AnnotationsRegOp(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %bazInit = firrtl.wire  : !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subindex %bazInit[0] : !firrtl.vector<uint<1>, 2>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = firrtl.subindex %bazInit[1] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %bar = firrtl.reg %clock  {annotations = [{a = "a"}], name = "bar"} : !firrtl.clock, !firrtl.vector<uint<1>, 2>
    %baz = firrtl.regreset %clock, %reset, %bazInit  {annotations = [{b = "b"}], name = "baz"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
  // CHECK: firrtl.reg
  // CHECK-SAME: annotations = [{a = "a"}]
  // CHECK: firrtl.reg
  // CHECK-SAME: annotations = [{a = "a"}]
  // CHECK: firrtl.regreset
  // CHECK-SAME: annotations = [{b = "b"}]
  // CHECK: firrtl.regreset
  // CHECK-SAME: annotations = [{b = "b"}]

// Test that WhenOp with regions has its regions lowered.
// CHECK-LABEL: firrtl.module private @WhenOp
  firrtl.module private @WhenOp (in %p: !firrtl.uint<1>,
                         in %in : !firrtl.bundle<a: uint<1>, b: uint<1>>,
                         out %out : !firrtl.bundle<a: uint<1>, b: uint<1>>) {
    // No else region.
    firrtl.when %p : !firrtl.uint<1> {
      // CHECK: firrtl.matchingconnect %out_a, %in_a : !firrtl.uint<1>
      // CHECK: firrtl.matchingconnect %out_b, %in_b : !firrtl.uint<1>
      firrtl.connect %out, %in : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
    }

    // Else region.
    firrtl.when %p : !firrtl.uint<1> {
      // CHECK: firrtl.matchingconnect %out_a, %in_a : !firrtl.uint<1>
      // CHECK: firrtl.matchingconnect %out_b, %in_b : !firrtl.uint<1>
      firrtl.connect %out, %in : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
    } else {
      // CHECK: firrtl.matchingconnect %out_a, %in_a : !firrtl.uint<1>
      // CHECK: firrtl.matchingconnect %out_b, %in_b : !firrtl.uint<1>
      firrtl.connect %out, %in : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
    }
  }

// Test that subfield annotations on wire are lowred to appropriate instance based on fieldID.
  // CHECK-LABEL: firrtl.module private @AnnotationsBundle
  firrtl.module private @AnnotationsBundle() {
    %bar = firrtl.wire  {annotations = [
      {circt.fieldID = 3, one},
      {circt.fieldID = 5, two}
    ]} : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>

      // TODO: Enable this
      // CHECK: %bar_0_baz = firrtl.wire  : !firrtl.uint<1>
      // CHECK: %bar_0_qux = firrtl.wire {annotations = [{one}]} : !firrtl.uint<1>
      // CHECK: %bar_1_baz = firrtl.wire {annotations = [{two}]} : !firrtl.uint<1>
      // CHECK: %bar_1_qux = firrtl.wire  : !firrtl.uint<1>

    %quux = firrtl.wire  {annotations = [
      {circt.fieldID = 0, zero}
    ]} : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
      // CHECK: %quux_0_baz = firrtl.wire {annotations = [{zero}]} : !firrtl.uint<1>
      // CHECK: %quux_0_qux = firrtl.wire {annotations = [{zero}]} : !firrtl.uint<1>
      // CHECK: %quux_1_baz = firrtl.wire {annotations = [{zero}]} : !firrtl.uint<1>
      // CHECK: %quux_1_qux = firrtl.wire {annotations = [{zero}]} : !firrtl.uint<1>
  }

// Test that subfield annotations on reg are lowred to appropriate instance based on fieldID.
 // CHECK-LABEL: firrtl.module private @AnnotationsBundle2
  firrtl.module private @AnnotationsBundle2(in %clock: !firrtl.clock) {
    %bar = firrtl.reg %clock  {annotations = [
      {circt.fieldID = 3, one},
      {circt.fieldID = 5, two}
    ]} : !firrtl.clock, !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>

    // TODO: Enable this
    // CHECK: %bar_0_baz = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_0_qux = firrtl.reg %clock  {annotations = [{one}]} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_1_baz = firrtl.reg %clock  {annotations = [{two}]} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_1_qux = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
  }

// Test that subfield annotations on reg are lowred to appropriate instance based on fieldID. Ignore un-flattened array targets
// circuit Foo: %[[{"one":null,"target":"~Foo|Foo>bar[0].qux[0]"},{"two":null,"target":"~Foo|Foo>bar[1].baz"},{"three":null,"target":"~Foo|Foo>bar[0].yes"} ]]

 // CHECK-LABEL: firrtl.module private @AnnotationsBundle3
  firrtl.module private @AnnotationsBundle3(in %clock: !firrtl.clock) {
    %bar = firrtl.reg %clock  {
      annotations = [
        {circt.fieldID = 6, one},
        {circt.fieldID = 12, two},
        {circt.fieldID = 8, three}
      ]} : !firrtl.clock, !firrtl.vector<bundle<baz: vector<uint<1>, 2>, qux: vector<uint<1>, 2>, yes: bundle<a: uint<1>, b: uint<1>>>, 2>

    // TODO: Enable this
    // CHECK: %bar_0_baz_0 = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_0_baz_1 = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_0_qux_0 = firrtl.reg %clock  {annotations = [{one}]} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_0_qux_1 = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_0_yes_a = firrtl.reg %clock  {annotations = [{three}]} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_0_yes_b = firrtl.reg %clock  {annotations = [{three}]} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_1_baz_0 = firrtl.reg %clock  {annotations = [{two}]} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_1_baz_1 = firrtl.reg %clock  {annotations = [{two}]} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_1_qux_0 = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_1_qux_1 = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_1_yes_a = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_1_yes_b = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
  }

// Test wire connection semantics.  Based on the flippedness of the destination
// type, the connection may be reversed.
// CHECK-LABEL: firrtl.module private @WireSemantics
  firrtl.module private @WireSemantics() {
    %a = firrtl.wire  : !firrtl.bundle<a: bundle<a: uint<1>>>
    %ax = firrtl.wire  : !firrtl.bundle<a: bundle<a: uint<1>>>
    // CHECK:  %a_a_a = firrtl.wire
    // CHECK-NEXT:  %ax_a_a = firrtl.wire
    firrtl.connect %a, %ax : !firrtl.bundle<a: bundle<a: uint<1>>>, !firrtl.bundle<a: bundle<a: uint<1>>>
    // a <= ax
    // CHECK-NEXT: firrtl.matchingconnect %a_a_a, %ax_a_a : !firrtl.uint<1>
    %0 = firrtl.subfield %a[a] : !firrtl.bundle<a: bundle<a: uint<1>>>
    %1 = firrtl.subfield %ax[a] : !firrtl.bundle<a: bundle<a: uint<1>>>
    firrtl.connect %0, %1 : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    // a.a <= ax.a
    // CHECK: firrtl.matchingconnect %a_a_a, %ax_a_a : !firrtl.uint<1>
    %2 = firrtl.subfield %a[a] : !firrtl.bundle<a: bundle<a: uint<1>>>
    %3 = firrtl.subfield %2[a] : !firrtl.bundle<a: uint<1>>
    %4 = firrtl.subfield %ax[a] : !firrtl.bundle<a: bundle<a: uint<1>>>
    %5 = firrtl.subfield %4[a] : !firrtl.bundle<a: uint<1>>
    firrtl.connect %3, %5 : !firrtl.uint<1>, !firrtl.uint<1>
    // a.a.a <= ax.a.a
    // CHECK: firrtl.connect %a_a_a, %ax_a_a
    %b = firrtl.wire  : !firrtl.bundle<a: bundle<a flip: uint<1>>>
    %bx = firrtl.wire  : !firrtl.bundle<a: bundle<a flip: uint<1>>>
    // CHECK: %b_a_a = firrtl.wire
    // CHECK: %bx_a_a = firrtl.wire
    firrtl.connect %b, %bx : !firrtl.bundle<a: bundle<a flip: uint<1>>>, !firrtl.bundle<a: bundle<a flip: uint<1>>>
    // b <= bx
    // CHECK: firrtl.matchingconnect %bx_a_a, %b_a_a
    %6 = firrtl.subfield %b[a] : !firrtl.bundle<a: bundle<a flip: uint<1>>>
    %7 = firrtl.subfield %bx[a] : !firrtl.bundle<a: bundle<a flip: uint<1>>>
    firrtl.connect %6, %7 : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>
    // b.a <= bx.a
    // CHECK: firrtl.matchingconnect %bx_a_a, %b_a_a
    %8 = firrtl.subfield %b[a] : !firrtl.bundle<a: bundle<a flip: uint<1>>>
    %9 = firrtl.subfield %8[a] : !firrtl.bundle<a flip: uint<1>>
    %10 = firrtl.subfield %bx[a] : !firrtl.bundle<a: bundle<a flip: uint<1>>>
    %11 = firrtl.subfield %10[a] : !firrtl.bundle<a flip: uint<1>>
    firrtl.connect %9, %11 : !firrtl.uint<1>, !firrtl.uint<1>
    // b.a.a <= bx.a.a
    // CHECK: firrtl.connect %b_a_a, %bx_a_a
    %c = firrtl.wire  : !firrtl.bundle<a flip: bundle<a: uint<1>>>
    %cx = firrtl.wire  : !firrtl.bundle<a flip: bundle<a: uint<1>>>
    // CHECK: %c_a_a = firrtl.wire : !firrtl.uint<1>
    // CHECK-NEXT: %cx_a_a = firrtl.wire : !firrtl.uint<1>
    firrtl.connect %c, %cx : !firrtl.bundle<a flip: bundle<a: uint<1>>>, !firrtl.bundle<a flip: bundle<a: uint<1>>>
    // c <= cx
    // CHECK: firrtl.matchingconnect %cx_a_a, %c_a_a
    %12 = firrtl.subfield %c[a] : !firrtl.bundle<a flip: bundle<a: uint<1>>>
    %13 = firrtl.subfield %cx[a] : !firrtl.bundle<a flip: bundle<a: uint<1>>>
    firrtl.connect %12, %13 : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    // c.a <= cx.a
    // CHECK: firrtl.matchingconnect %c_a_a, %cx_a_a
    %14 = firrtl.subfield %c[a] : !firrtl.bundle<a flip: bundle<a: uint<1>>>
    %15 = firrtl.subfield %14[a] : !firrtl.bundle<a: uint<1>>
    %16 = firrtl.subfield %cx[a] : !firrtl.bundle<a flip: bundle<a: uint<1>>>
    %17 = firrtl.subfield %16[a] : !firrtl.bundle<a: uint<1>>
    firrtl.connect %15, %17 : !firrtl.uint<1>, !firrtl.uint<1>
    // c.a.a <= cx.a.a
    // CHECK: firrtl.connect %c_a_a, %cx_a_a
    %d = firrtl.wire  : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    %dx = firrtl.wire  : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    // CHECK: %d_a_a = firrtl.wire : !firrtl.uint<1>
    // CHECK-NEXT: %dx_a_a = firrtl.wire : !firrtl.uint<1>
    firrtl.connect %d, %dx : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>, !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    // d <= dx
    // CHECK: firrtl.matchingconnect %d_a_a, %dx_a_a
    %18 = firrtl.subfield %d[a] : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    %19 = firrtl.subfield %dx[a] : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    firrtl.connect %18, %19 : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>
    // d.a <= dx.a
    // CHECK: firrtl.matchingconnect %dx_a_a, %d_a_a
    %20 = firrtl.subfield %d[a] : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    %21 = firrtl.subfield %20[a] : !firrtl.bundle<a flip: uint<1>>
    %22 = firrtl.subfield %dx[a] : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    %23 = firrtl.subfield %22[a] : !firrtl.bundle<a flip: uint<1>>
    firrtl.connect %21, %23 : !firrtl.uint<1>, !firrtl.uint<1>
    // d.a.a <= dx.a.a
    // CHECK: firrtl.connect %d_a_a, %dx_a_a
  }

// Test that a vector of bundles with a write works.
 // CHECK-LABEL: firrtl.module private @aofs
    firrtl.module private @aofs(in %a: !firrtl.uint<1>, in %sel: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>>, 4>) {
      %0 = firrtl.subindex %b[0] : !firrtl.vector<bundle<wo: uint<1>>, 4>
      %1 = firrtl.subfield %0[wo] : !firrtl.bundle<wo: uint<1>>
      %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
      firrtl.connect %1, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
      %2 = firrtl.subindex %b[1] : !firrtl.vector<bundle<wo: uint<1>>, 4>
      %3 = firrtl.subfield %2[wo] : !firrtl.bundle<wo: uint<1>>
      %invalid_ui1_0 = firrtl.invalidvalue : !firrtl.uint<1>
      firrtl.connect %3, %invalid_ui1_0 : !firrtl.uint<1>, !firrtl.uint<1>
      %4 = firrtl.subindex %b[2] : !firrtl.vector<bundle<wo: uint<1>>, 4>
      %5 = firrtl.subfield %4[wo] : !firrtl.bundle<wo: uint<1>>
      %invalid_ui1_1 = firrtl.invalidvalue : !firrtl.uint<1>
      firrtl.connect %5, %invalid_ui1_1 : !firrtl.uint<1>, !firrtl.uint<1>
      %6 = firrtl.subindex %b[3] : !firrtl.vector<bundle<wo: uint<1>>, 4>
      %7 = firrtl.subfield %6[wo] : !firrtl.bundle<wo: uint<1>>
      %invalid_ui1_2 = firrtl.invalidvalue : !firrtl.uint<1>
      firrtl.connect %7, %invalid_ui1_2 : !firrtl.uint<1>, !firrtl.uint<1>
      %8 = firrtl.subaccess %b[%sel] : !firrtl.vector<bundle<wo: uint<1>>, 4>, !firrtl.uint<2>
      %9 = firrtl.subfield %8[wo] : !firrtl.bundle<wo: uint<1>>
      firrtl.connect %9, %a : !firrtl.uint<1>, !firrtl.uint<1>
    }


// Test that annotations on aggregate ports are copied.
  firrtl.extmodule @Sub1(in a: !firrtl.vector<uint<1>, 2> [{a}])
  // CHECK-LABEL: firrtl.extmodule @Sub1
  // CHECK-COUNT-2: [{b}]
  // CHECK-NOT: [{a}]
  firrtl.module private @Port(in %a: !firrtl.vector<uint<1>, 2> [{b}]) {
    %sub_a = firrtl.instance sub @Sub1(in a: !firrtl.vector<uint<1>, 2>)
    firrtl.connect %sub_a, %a : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }

// Test that annotations on subfield/subindices of ports are only applied to
// matching fieldIDs.
    // The annotation should be copied to just a.a.  The firrtl.hello arg
    // attribute should be copied to each new port.
    firrtl.module private @PortBundle(in %a: !firrtl.bundle<a: uint<1>, b flip: uint<1>> [{circt.fieldID = 1, a}]) {}
    // CHECK-LABEL: firrtl.module private @PortBundle
    // CHECK-SAME:    in %a_a: !firrtl.uint<1> [{a}]

// circuit Foo:
//   module Foo:
//     input a: UInt<2>[2][2]
//     input sel: UInt<2>
//     output b: UInt<2>
//
//     b <= a[sel][sel]

  firrtl.module private @multidimRead(
    in %a: !firrtl.vector<vector<uint<2>, 2>, 2>,
    in %sel: !firrtl.uint<2>,
    out %b: !firrtl.uint<2>) {
    %0 = firrtl.subaccess %a[%sel] : !firrtl.vector<vector<uint<2>, 2>, 2>, !firrtl.uint<2>
    %1 = firrtl.subaccess %0[%sel] : !firrtl.vector<uint<2>, 2>, !firrtl.uint<2>
    firrtl.connect %b, %1 : !firrtl.uint<2>, !firrtl.uint<2>
  }

// CHECK-LABEL: firrtl.module private @multidimRead(in %a_0_0: !firrtl.uint<2>, in %a_0_1: !firrtl.uint<2>, in %a_1_0: !firrtl.uint<2>, in %a_1_1: !firrtl.uint<2>, in %sel: !firrtl.uint<2>, out %b: !firrtl.uint<2>) {
// CHECK-NEXT:      %0 = firrtl.multibit_mux %sel, %a_1_0, %a_0_0 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT:      %1 = firrtl.multibit_mux %sel, %a_1_1, %a_0_1 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT:      %2 = firrtl.multibit_mux %sel, %1, %0 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT:      firrtl.connect %b, %2 : !firrtl.uint<2>
// CHECK-NEXT: }

//  module Foo:
//    input b: UInt<1>
//    input sel: UInt<2>
//    input default: UInt<1>[4]
//    output a: UInt<1>[4]
//
//     a <= default
//     a[sel] <= b

  firrtl.module private @write1D(in %b: !firrtl.uint<1>, in %sel: !firrtl.uint<2>, in %default: !firrtl.vector<uint<1>, 2>, out %a: !firrtl.vector<uint<1>, 2>) {
    firrtl.connect %a, %default : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subaccess %a[%sel] : !firrtl.vector<uint<1>, 2>, !firrtl.uint<2>
    firrtl.connect %0, %b : !firrtl.uint<1>, !firrtl.uint<1>
  }
// CHECK-LABEL:    firrtl.module private @write1D(in %b: !firrtl.uint<1>, in %sel: !firrtl.uint<2>, in %default_0: !firrtl.uint<1>, in %default_1: !firrtl.uint<1>, out %a_0: !firrtl.uint<1>, out %a_1: !firrtl.uint<1>) {
// CHECK-NEXT:      firrtl.matchingconnect %a_0, %default_0 : !firrtl.uint<1>
// CHECK-NEXT:      firrtl.matchingconnect %a_1, %default_1 : !firrtl.uint<1>
// CHECK-NEXT:      %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
// CHECK-NEXT:      %0 = firrtl.eq %sel, %c0_ui1 : (!firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:      firrtl.when %0 : !firrtl.uint<1> {
// CHECK-NEXT:        firrtl.matchingconnect %a_0, %b : !firrtl.uint<1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
// CHECK-NEXT:      %1 = firrtl.eq %sel, %c1_ui1 : (!firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:      firrtl.when %1 : !firrtl.uint<1> {
// CHECK-NEXT:        firrtl.matchingconnect %a_1, %b : !firrtl.uint<1>
// CHECK-NEXT:      }
// CHECK-NEXT:    }


// circuit Foo:
//   module Foo:
//     input sel: UInt<1>
//     input b: UInt<2>
//     output a: UInt<2>[2][2]
//
//     a[sel][sel] <= b

  firrtl.module private @multidimWrite(in %sel: !firrtl.uint<1>, in %b: !firrtl.uint<2>, out %a: !firrtl.vector<vector<uint<2>, 2>, 2>) {
    %0 = firrtl.subaccess %a[%sel] : !firrtl.vector<vector<uint<2>, 2>, 2>, !firrtl.uint<1>
    %1 = firrtl.subaccess %0[%sel] : !firrtl.vector<uint<2>, 2>, !firrtl.uint<1>
    firrtl.connect %1, %b : !firrtl.uint<2>, !firrtl.uint<2>
  }
// CHECK-LABEL:    firrtl.module private @multidimWrite(in %sel: !firrtl.uint<1>, in %b: !firrtl.uint<2>, out %a_0_0: !firrtl.uint<2>, out %a_0_1: !firrtl.uint<2>, out %a_1_0: !firrtl.uint<2>, out %a_1_1: !firrtl.uint<2>) {
// CHECK-NEXT:      %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
// CHECK-NEXT:      %0 = firrtl.eq %sel, %c0_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:      firrtl.when %0 : !firrtl.uint<1> {
// CHECK-NEXT:        %c0_ui1_0 = firrtl.constant 0 : !firrtl.uint<1>
// CHECK-NEXT:        %2 = firrtl.eq %sel, %c0_ui1_0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:        firrtl.when %2 : !firrtl.uint<1> {
// CHECK-NEXT:          firrtl.matchingconnect %a_0_0, %b : !firrtl.uint<2>
// CHECK-NEXT:        }
// CHECK-NEXT:        %c1_ui1_1 = firrtl.constant 1 : !firrtl.uint<1>
// CHECK-NEXT:        %3 = firrtl.eq %sel, %c1_ui1_1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:        firrtl.when %3 : !firrtl.uint<1> {
// CHECK-NEXT:          firrtl.matchingconnect %a_0_1, %b : !firrtl.uint<2>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
// CHECK-NEXT:      %1 = firrtl.eq %sel, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:      firrtl.when %1 : !firrtl.uint<1> {
// CHECK-NEXT:        %c0_ui1_0 = firrtl.constant 0 : !firrtl.uint<1>
// CHECK-NEXT:        %2 = firrtl.eq %sel, %c0_ui1_0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:        firrtl.when %2 : !firrtl.uint<1> {
// CHECK-NEXT:          firrtl.matchingconnect %a_1_0, %b : !firrtl.uint<2>
// CHECK-NEXT:        }
// CHECK-NEXT:        %c1_ui1_1 = firrtl.constant 1 : !firrtl.uint<1>
// CHECK-NEXT:        %3 = firrtl.eq %sel, %c1_ui1_1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:        firrtl.when %3 : !firrtl.uint<1> {
// CHECK-NEXT:          firrtl.matchingconnect %a_1_1, %b : !firrtl.uint<2>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// circuit Foo:
//   module Foo:
//     input a: {wo: UInt<1>, valid: UInt<2>}
//     input def: {wo: UInt<1>, valid: UInt<2>}[4]
//     input sel: UInt<2>
//     output b: {wo: UInt<1>, valid: UInt<2>}[4]
//
//     b <= def
//     b[sel].wo <= a.wo
  firrtl.module private @writeVectorOfBundle1D(in %a: !firrtl.bundle<wo: uint<1>, valid: uint<2>>, in %def: !firrtl.vector<bundle<wo: uint<1>, valid: uint<2>>, 2>, in %sel: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>, valid: uint<2>>, 2>) {
    firrtl.connect %b, %def : !firrtl.vector<bundle<wo: uint<1>, valid: uint<2>>, 2>, !firrtl.vector<bundle<wo: uint<1>, valid: uint<2>>, 2>
    %0 = firrtl.subaccess %b[%sel] : !firrtl.vector<bundle<wo: uint<1>, valid: uint<2>>, 2>, !firrtl.uint<2>
    %1 = firrtl.subfield %0[wo] : !firrtl.bundle<wo: uint<1>, valid: uint<2>>
    %2 = firrtl.subfield %a[wo] : !firrtl.bundle<wo: uint<1>, valid: uint<2>>
    firrtl.connect %1, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }

// CHECK-LABEL:    firrtl.module private @writeVectorOfBundle1D(in %a_wo: !firrtl.uint<1>, in %a_valid: !firrtl.uint<2>, in %def_0_wo: !firrtl.uint<1>, in %def_0_valid: !firrtl.uint<2>, in %def_1_wo: !firrtl.uint<1>, in %def_1_valid: !firrtl.uint<2>, in %sel: !firrtl.uint<2>, out %b_0_wo: !firrtl.uint<1>, out %b_0_valid: !firrtl.uint<2>, out %b_1_wo: !firrtl.uint<1>, out %b_1_valid: !firrtl.uint<2>) {
// CHECK-NEXT:      firrtl.matchingconnect %b_0_wo, %def_0_wo : !firrtl.uint<1>
// CHECK-NEXT:      firrtl.matchingconnect %b_0_valid, %def_0_valid : !firrtl.uint<2>
// CHECK-NEXT:      firrtl.matchingconnect %b_1_wo, %def_1_wo : !firrtl.uint<1>
// CHECK-NEXT:      firrtl.matchingconnect %b_1_valid, %def_1_valid : !firrtl.uint<2>
// CHECK-NEXT:      %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
// CHECK-NEXT:      %0 = firrtl.eq %sel, %c0_ui1 : (!firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:      firrtl.when %0 : !firrtl.uint<1> {
// CHECK-NEXT:        firrtl.matchingconnect %b_0_wo, %a_wo : !firrtl.uint<1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
// CHECK-NEXT:      %1 = firrtl.eq %sel, %c1_ui1 : (!firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:      firrtl.when %1 : !firrtl.uint<1> {
// CHECK-NEXT:        firrtl.matchingconnect %b_1_wo, %a_wo : !firrtl.uint<1>
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// circuit Foo:
//   module Foo:
//     input a: UInt<2>[2][2]
//     input sel1: UInt<1>
//     input sel2: UInt<1>
//     output b: UInt<2>
//     output c: UInt<2>
//
//     b <= a[sel1][sel1]
//     c <= a[sel1][sel2]
  firrtl.module private @multiSubaccess(in %a: !firrtl.vector<vector<uint<2>, 2>, 2>, in %sel1: !firrtl.uint<1>, in %sel2: !firrtl.uint<1>, out %b: !firrtl.uint<2>, out %c: !firrtl.uint<2>) {
    %0 = firrtl.subaccess %a[%sel1] : !firrtl.vector<vector<uint<2>, 2>, 2>, !firrtl.uint<1>
    %1 = firrtl.subaccess %0[%sel1] : !firrtl.vector<uint<2>, 2>, !firrtl.uint<1>
    firrtl.connect %b, %1 : !firrtl.uint<2>, !firrtl.uint<2>
    %2 = firrtl.subaccess %a[%sel1] : !firrtl.vector<vector<uint<2>, 2>, 2>, !firrtl.uint<1>
    %3 = firrtl.subaccess %2[%sel2] : !firrtl.vector<uint<2>, 2>, !firrtl.uint<1>
    firrtl.connect %c, %3 : !firrtl.uint<2>, !firrtl.uint<2>
  }

// CHECK-LABEL:    firrtl.module private @multiSubaccess(in %a_0_0: !firrtl.uint<2>, in %a_0_1: !firrtl.uint<2>, in %a_1_0: !firrtl.uint<2>, in %a_1_1: !firrtl.uint<2>, in %sel1: !firrtl.uint<1>, in %sel2: !firrtl.uint<1>, out %b: !firrtl.uint<2>, out %c: !firrtl.uint<2>) {
// CHECK-NEXT:      %0 = firrtl.multibit_mux %sel1, %a_1_0, %a_0_0 : !firrtl.uint<1>, !firrtl.uint<2>
// CHECK-NEXT:      %1 = firrtl.multibit_mux %sel1, %a_1_1, %a_0_1 : !firrtl.uint<1>, !firrtl.uint<2>
// CHECK-NEXT:      %2 = firrtl.multibit_mux %sel1, %1, %0 : !firrtl.uint<1>, !firrtl.uint<2>
// CHECK-NEXT:      firrtl.connect %b, %2 : !firrtl.uint<2>
// CHECK-NEXT:      %3 = firrtl.multibit_mux %sel1, %a_1_0, %a_0_0 : !firrtl.uint<1>, !firrtl.uint<2>
// CHECK-NEXT:      %4 = firrtl.multibit_mux %sel1, %a_1_1, %a_0_1 : !firrtl.uint<1>, !firrtl.uint<2>
// CHECK-NEXT:      %5 = firrtl.multibit_mux %sel2, %4, %3 : !firrtl.uint<1>, !firrtl.uint<2>
// CHECK-NEXT:      firrtl.connect %c, %5 : !firrtl.uint<2>
// CHECK-NEXT:    }


// Handle zero-length vector subaccess
  // CHECK-LABEL: zvec
  firrtl.module private @zvec(
      in %i: !firrtl.vector<bundle<a: uint<8>, b: uint<4>>, 0>,
      in %sel: !firrtl.uint<1>,
      out %foo: !firrtl.vector<uint<1>, 0>,
      out %o: !firrtl.uint<8>)
    {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.subaccess %foo[%c0_ui1] : !firrtl.vector<uint<1>, 0>, !firrtl.uint<1>
    firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = firrtl.subaccess %i[%sel] : !firrtl.vector<bundle<a: uint<8>, b: uint<4>>, 0>, !firrtl.uint<1>
    %2 = firrtl.subfield %1[a] : !firrtl.bundle<a: uint<8>, b: uint<4>>
    firrtl.connect %o, %2 : !firrtl.uint<8>, !firrtl.uint<8>
  // CHECK: firrtl.connect %o, %invalid_ui8
  }

// Test InstanceOp with port annotations.
// CHECK-LABEL: firrtl.module private @Bar3
  firrtl.module private @Bar3(in %a: !firrtl.uint<1>, out %b: !firrtl.bundle<baz: uint<1>, qux: uint<1>>) {
  }
  // CHECK-LABEL: firrtl.module private @Foo3
  firrtl.module private @Foo3() {
    // CHECK: in a: !firrtl.uint<1> [{one}], out b_baz: !firrtl.uint<1> [{two}], out b_qux: !firrtl.uint<1>
    %bar_a, %bar_b = firrtl.instance bar @Bar3(
      in a: !firrtl.uint<1> [{one}],
      out b: !firrtl.bundle<baz: uint<1>, qux: uint<1>> [{circt.fieldID = 1, two}]
    )
  }


// Test MemOp with port annotations.
// circuit Foo: %[[{"a":null,"target":"~Foo|Foo>bar.r"},
//                 {"b":null,"target":"~Foo|Foo>bar.r.data"},
//                 {"c":null,"target":"~Foo|Foo>bar.w.en"},
//                 {"d":null,"target":"~Foo|Foo>bar.w.data.qux"},
//                 {"e":null,"target":"~Foo|Foo>bar.rw.wmode"}
//                 {"f":null,"target":"~Foo|Foo>bar.rw.wmask.baz"}]]

// CHECK-LABEL: firrtl.module private @Foo4
  firrtl.module private @Foo4() {
    // CHECK: firrtl.mem
    // CHECK-SAME: portAnnotations = [
    // CHECK-SAME: [{a}, {b, circt.fieldID = 4 : i32}],
    // CHECK-SAME: [{c, circt.fieldID = 2 : i32}]
    // CHECK-SAME: [{circt.fieldID = 4 : i32, e}, {circt.fieldID = 7 : i32, f}]

    // CHECK: firrtl.mem
    // CHECK-SAME: portAnnotations = [
    // CHECK-SAME: [{a}, {b, circt.fieldID = 4 : i32}],
    // CHECK-SAME: [{c, circt.fieldID = 2 : i32}, {circt.fieldID = 4 : i32, d}]
    // CHECK-SAME: [{circt.fieldID = 4 : i32, e}]

    %bar_r, %bar_w, %bar_rw = firrtl.mem Undefined  {depth = 16 : i64, name = "bar",
        portAnnotations = [
          [{a}, {circt.fieldID = 4 : i32, b}],
          [{circt.fieldID = 2 : i32, c}, {circt.fieldID = 6 : i32, d}],
          [{circt.fieldID = 4 : i32, e}, {circt.fieldID = 12 : i32, f}]
        ],
        portNames = ["r", "w", "rw"], readLatency = 0 : i32, writeLatency = 1 : i32} :
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<baz: uint<8>, qux: uint<8>>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<baz: uint<8>, qux: uint<8>>, mask: bundle<baz: uint<1>, qux: uint<1>>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<baz: uint<8>, qux: uint<8>>, wmode: uint<1>, wdata: bundle<baz: uint<8>, qux: uint<8>>, wmask: bundle<baz: uint<1>, qux: uint<1>>>
  }

// This simply has to not crash
// CHECK-LABEL: firrtl.module private @vecmem
firrtl.module private @vecmem(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
  %vmem_MPORT, %vmem_rdwrPort = firrtl.mem Undefined  {depth = 32 : i64, name = "vmem", portNames = ["MPORT", "rdwrPort"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data flip: vector<uint<17>, 8>>, !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, rdata flip: vector<uint<17>, 8>, wmode: uint<1>, wdata: vector<uint<17>, 8>, wmask: vector<uint<1>, 8>>
}

// Issue 1436
firrtl.extmodule @is1436_BAR(out io: !firrtl.bundle<llWakeup flip: vector<uint<1>, 1>>)
// CHECK-LABEL: firrtl.module private @is1436_FOO
firrtl.module private @is1436_FOO() {
  %thing_io = firrtl.instance thing @is1436_BAR(out io: !firrtl.bundle<llWakeup flip: vector<uint<1>, 1>>)
  %0 = firrtl.subfield %thing_io[llWakeup] : !firrtl.bundle<llWakeup flip: vector<uint<1>, 1>>
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %1 = firrtl.subaccess %0[%c0_ui2] : !firrtl.vector<uint<1>, 1>, !firrtl.uint<2>
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  firrtl.connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module private @BitCast1
  firrtl.module private @BitCast1(out %o_0: !firrtl.uint<1>, out %o_1: !firrtl.uint<1>) {
    %a1 = firrtl.wire : !firrtl.uint<4>
    %b = firrtl.bitcast %a1 : (!firrtl.uint<4>) -> (!firrtl.vector<uint<2>, 2>)
    // CHECK:  %[[v0:.+]] = firrtl.bits %a1 1 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<2>
    // CHECK-NEXT:  %[[v1:.+]] = firrtl.bits %a1 3 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<2>
    // CHECK-NEXT:  %[[v2:.+]] = firrtl.cat %[[v1]], %[[v0]] : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<4>
    %c = firrtl.bitcast %b : (!firrtl.vector<uint<2>, 2>) -> (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<2>>)
    // CHECK-NEXT:  %[[v3:.+]] = firrtl.bits %[[v2]] 3 to 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT:  %[[v4:.+]] = firrtl.bits %[[v2]] 2 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT:  %[[v5:.+]] = firrtl.bits %[[v2]] 1 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<2>
    %d = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<2>>
    // CHECK-NEXT:  %d_valid = firrtl.wire  : !firrtl.uint<1>
    // CHECK-NEXT:  %d_ready = firrtl.wire  : !firrtl.uint<1>
    // CHECK-NEXT:  %d_data = firrtl.wire  : !firrtl.uint<2>
    firrtl.connect %d , %c: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<2>>, !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<2>>
    // CHECK-NEXT:  firrtl.matchingconnect %d_valid, %[[v3]] : !firrtl.uint<1>
    // CHECK-NEXT:  firrtl.matchingconnect %d_ready, %[[v4]] : !firrtl.uint<1>
    // CHECK-NEXT:  firrtl.matchingconnect %d_data, %[[v5]] : !firrtl.uint<2>
    %e = firrtl.bitcast %d : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<2>>) -> (!firrtl.bundle<addr: uint<2>, data : vector<uint<1>, 2>>)
    // CHECK-NEXT:  %[[v6:.+]] = firrtl.cat %d_valid, %d_ready : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    // CHECK-NEXT:  %[[v7:.+]] = firrtl.cat %[[v6]], %d_data : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<4>
    // CHECK-NEXT:  %[[v8:.+]] = firrtl.bits %[[v7]] 3 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<2>
    // CHECK-NEXT:  %[[v9:.+]] = firrtl.bits %[[v7]] 1 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  %o1 = firrtl.subfield %e[data] : !firrtl.bundle<addr: uint<2>, data : vector<uint<1>, 2>>
   %c2 = firrtl.bitcast %a1 : (!firrtl.uint<4>) -> (!firrtl.bundle<valid: bundle<re: bundle<a: uint<1>>, aa: uint<1>>, ready: uint<1>, data: uint<1>>)
    %d2 = firrtl.wire : !firrtl.bundle<valid: bundle<re: bundle<a: uint<1>>, aa: uint<1>>, ready: uint<1>, data: uint<1>>
    firrtl.connect %d2 , %c2: !firrtl.bundle<valid: bundle<re: bundle<a: uint<1>>, aa: uint<1>>, ready: uint<1>, data:
    uint<1>>, !firrtl.bundle<valid: bundle<re: bundle<a: uint<1>>, aa: uint<1>>, ready: uint<1>, data: uint<1>>
   //CHECK: %[[v10:.+]] = firrtl.bits %[[v9]] 0 to 0 : (!firrtl.uint<2>) -> !firrtl.uint<1>
   //CHECK: %[[v11:.+]] = firrtl.bits %[[v9]] 1 to 1 : (!firrtl.uint<2>) -> !firrtl.uint<1>
   //CHECK: %[[v12:.+]] = firrtl.bits %a1 3 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<2>
   //CHECK: %[[v13:.+]] = firrtl.bits %[[v12]] 1 to 1 : (!firrtl.uint<2>) -> !firrtl.uint<1>
   //CHECK: %[[v14:.+]] = firrtl.bits %[[v13]] 0 to 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
   //CHECK: %[[v15:.+]] = firrtl.bits %[[v12]] 0 to 0 : (!firrtl.uint<2>) -> !firrtl.uint<1>
   //CHECK: %[[v16:.+]] = firrtl.bits %a1 1 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<1>
   //CHECK: %[[v17:.+]] = firrtl.bits %a1 0 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<1>
   //CHECK: %[[d2_valid_re_a:.+]] = firrtl.wire  : !firrtl.uint<1>
   //CHECK: %[[d2_valid_aa:.+]] = firrtl.wire  : !firrtl.uint<1>
   //CHECK: %[[d2_ready:.+]] = firrtl.wire  : !firrtl.uint<1>
   //CHECK: %[[d2_data:.+]] = firrtl.wire  : !firrtl.uint<1>
   //CHECK: firrtl.matchingconnect %[[d2_valid_re_a]], %[[v14]] : !firrtl.uint<1>
   //CHECK: firrtl.matchingconnect %[[d2_valid_aa]], %[[v15]] : !firrtl.uint<1>
   //CHECK: firrtl.matchingconnect %[[d2_ready]], %[[v16]] : !firrtl.uint<1>
   //CHECK: firrtl.matchingconnect %[[d2_data]], %[[v17]] : !firrtl.uint<1>

  }

  // Issue #2315: Narrow index constants overflow when subaccessing long vectors.
  // https://github.com/llvm/circt/issues/2315
  // CHECK-LABEL: firrtl.module private @Issue2315
  firrtl.module private @Issue2315(in %x: !firrtl.vector<uint<10>, 5>, in %source: !firrtl.uint<2>, out %z: !firrtl.uint<10>) {
    %0 = firrtl.subaccess %x[%source] : !firrtl.vector<uint<10>, 5>, !firrtl.uint<2>
    firrtl.connect %z, %0 : !firrtl.uint<10>, !firrtl.uint<10>
    // The width of multibit mux index will be converted at LowerToHW,
    // so it is ok that the type of `%source` is uint<2> here.
    // CHECK:      %0 = firrtl.multibit_mux %source, %x_4, %x_3, %x_2, %x_1, %x_0 : !firrtl.uint<2>, !firrtl.uint<10>
    // CHECK-NEXT: firrtl.connect %z, %0 : !firrtl.uint<10>
  }

  firrtl.module private @SendRefTypeBundles1(in %source: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>, out %sink: !firrtl.probe<bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>>) {
    // CHECK:  firrtl.module private @SendRefTypeBundles1(
    // CHECK-SAME:  in %source_valid: !firrtl.uint<1>,
    // CHECK-SAME:  in %source_ready: !firrtl.uint<1>,
    // CHECK-SAME:  in %source_data: !firrtl.uint<64>,
    // CHECK-SAME:  out %sink_valid: !firrtl.probe<uint<1>>,
    // CHECK-SAME:  out %sink_ready: !firrtl.probe<uint<1>>,
    // CHECK-SAME:  out %sink_data: !firrtl.probe<uint<64>>) {
    %0 = firrtl.ref.send %source : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
    // CHECK:  %0 = firrtl.ref.send %source_valid : !firrtl.uint<1>
    // CHECK:  %1 = firrtl.ref.send %source_ready : !firrtl.uint<1>
    // CHECK:  %2 = firrtl.ref.send %source_data : !firrtl.uint<64>
    firrtl.ref.define %sink, %0 : !firrtl.probe<bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>>
    // CHECK:  firrtl.ref.define %sink_valid, %0 : !firrtl.probe<uint<1>>
    // CHECK:  firrtl.ref.define %sink_ready, %1 : !firrtl.probe<uint<1>>
    // CHECK:  firrtl.ref.define %sink_data, %2 : !firrtl.probe<uint<64>>
  }
  firrtl.module private @SendRefTypeVectors1(in %a: !firrtl.vector<uint<1>, 2>, out %b: !firrtl.probe<vector<uint<1>, 2>>) {
    // CHECK-LABEL: firrtl.module private @SendRefTypeVectors1
    // CHECK-SAME: in %a_0: !firrtl.uint<1>, in %a_1: !firrtl.uint<1>, out %b_0: !firrtl.probe<uint<1>>, out %b_1: !firrtl.probe<uint<1>>)
    %0 = firrtl.ref.send %a : !firrtl.vector<uint<1>, 2>
    // CHECK:  %0 = firrtl.ref.send %a_0 : !firrtl.uint<1>
    // CHECK:  %1 = firrtl.ref.send %a_1 : !firrtl.uint<1>
    firrtl.ref.define %b, %0 : !firrtl.probe<vector<uint<1>, 2>>
    // CHECK:  firrtl.ref.define %b_0, %0 : !firrtl.probe<uint<1>>
    // CHECK:  firrtl.ref.define %b_1, %1 : !firrtl.probe<uint<1>>
  }
  firrtl.module private @RefTypeBundles2() {
    %x = firrtl.wire   : !firrtl.bundle<a: uint<1>, b: uint<2>>
    %0 = firrtl.ref.send %x : !firrtl.bundle<a: uint<1>, b: uint<2>>
    // CHECK:   %0 = firrtl.ref.send %x_a : !firrtl.uint<1>
    // CHECK:   %1 = firrtl.ref.send %x_b : !firrtl.uint<2>
    %1 = firrtl.ref.resolve %0 : !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>
    // CHECK:   %2 = firrtl.ref.resolve %0 : !firrtl.probe<uint<1>>
    // CHECK:   %3 = firrtl.ref.resolve %1 : !firrtl.probe<uint<2>>
  }
  firrtl.module private @RefTypeVectors(out %c: !firrtl.vector<uint<1>, 2>) {
    %x = firrtl.wire   : !firrtl.vector<uint<1>, 2>
    %0 = firrtl.ref.send %x : !firrtl.vector<uint<1>, 2>
    // CHECK:  %0 = firrtl.ref.send %x_0 : !firrtl.uint<1>
    // CHECK:  %1 = firrtl.ref.send %x_1 : !firrtl.uint<1>
    %1 = firrtl.ref.resolve %0 : !firrtl.probe<vector<uint<1>, 2>>
    // CHECK:  %2 = firrtl.ref.resolve %0 : !firrtl.probe<uint<1>>
    // CHECK:  %3 = firrtl.ref.resolve %1 : !firrtl.probe<uint<1>>
    firrtl.matchingconnect %c, %1 : !firrtl.vector<uint<1>, 2>
    // CHECK:  firrtl.matchingconnect %c_0, %2 : !firrtl.uint<1>
    // CHECK:  firrtl.matchingconnect %c_1, %3 : !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module private @RefTypeBV_RW
  firrtl.module private @RefTypeBV_RW(
    // CHECK-SAME: rwprobe<vector<uint<1>, 2>>
    out %vec_ref: !firrtl.rwprobe<vector<uint<1>, 2>>,
    // CHECK-NOT: firrtl.vector
    out %vec: !firrtl.vector<uint<1>, 2>,
    // CHECK-SAME: rwprobe<bundle
    out %bov_ref: !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>,
    // CHECK-NOT: bundle
    out %bov: !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>,
    // CHECK: firrtl.probe
    out %probe: !firrtl.probe<uint<2>>
  ) {
    // Forceable declaration are never expanded into ground elements.
    // CHECK-NEXT: %{{.+}}, %[[X_REF:.+]] = firrtl.wire forceable : !firrtl.bundle<a: vector<uint<1>, 2>, b flip: uint<2>>, !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>
    %x, %x_ref = firrtl.wire forceable : !firrtl.bundle<a: vector<uint<1>, 2>, b flip: uint<2>>, !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>

    // Define using forceable ref preserved.
    // CHECK-NEXT: firrtl.ref.define %{{.+}}, %[[X_REF]] : !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>
    firrtl.ref.define %bov_ref, %x_ref : !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>

    // Preserve ref.sub uses.
    // CHECK-NEXT: %[[X_REF_A:.+]] = firrtl.ref.sub %[[X_REF]][0]
    // CHECK-NEXT: %[[X_A:.+]] = firrtl.ref.resolve %[[X_REF_A]]
    // CHECK-NEXT: %[[v_0:.+]] = firrtl.subindex %[[X_A]][0]
    // CHECK-NEXT: firrtl.matchingconnect %vec_0, %[[v_0]]
    // CHECK-NEXT: %[[v_1:.+]] = firrtl.subindex %[[X_A]][1]
    // CHECK-NEXT: firrtl.matchingconnect %vec_1, %[[v_1]]
    %x_ref_a = firrtl.ref.sub %x_ref[0] : !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>
    %x_a = firrtl.ref.resolve %x_ref_a : !firrtl.rwprobe<vector<uint<1>, 2>>
    firrtl.matchingconnect %vec, %x_a : !firrtl.vector<uint<1>, 2>

    // Check chained ref.sub's work.
    // CHECK-NEXT: %[[X_A_1_REF:.+]] = firrtl.ref.sub %[[X_REF_A]][1]
    // CHECK-NEXT: firrtl.ref.resolve %[[X_A_1_REF]]
    %x_ref_a_1 = firrtl.ref.sub %x_ref_a[1] : !firrtl.rwprobe<vector<uint<1>, 2>>
    %x_a_1 = firrtl.ref.resolve %x_ref_a_1 : !firrtl.rwprobe<uint<1>>

    // Ref to flipped field.
    // CHECK-NEXT: %[[X_B_REF:.+]] = firrtl.ref.sub %[[X_REF]][1]
    // CHECK-NEXT: firrtl.ref.resolve %[[X_B_REF]]
    %x_ref_b = firrtl.ref.sub %x_ref[1] : !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>
    %x_b = firrtl.ref.resolve %x_ref_b : !firrtl.rwprobe<uint<2>>

    // rwprobe -> probe demotion.
    // CHECK-NEXT: %[[X_B_REF_PROBE:.+]] = firrtl.ref.cast %[[X_B_REF]]
    // CHECK-NEXT: firrtl.ref.define %probe, %[[X_B_REF_PROBE]] : !firrtl.probe<uint<2>>
    %x_ref_b_probe = firrtl.ref.cast %x_ref_b : (!firrtl.rwprobe<uint<2>>) -> !firrtl.probe<uint<2>>
    firrtl.ref.define %probe, %x_ref_b_probe : !firrtl.probe<uint<2>>

    // Aggregate ref.cast, check split up properly.  Demote to probe, cast away some width information.
    // (Note: The variables here override values used above)
    // CHECK-NEXT: %[[X_REF_A:.+]] = firrtl.ref.sub %[[X_REF]][0]
    // CHECK-NEXT: %[[X_REF_A_0:.+]] = firrtl.ref.sub %[[X_REF_A]][0]
    // CHECK-NEXT: %[[X_REF_A_0_CAST:.+]] = firrtl.ref.cast %[[X_REF_A_0]] : (!firrtl.rwprobe<uint<1>>) -> !firrtl.probe<uint>
    // CHECK-NEXT: %[[X_REF_A_1:.+]] = firrtl.ref.sub %[[X_REF_A]][1]
    // CHECK-NEXT: %[[X_REF_A_1_CAST:.+]] = firrtl.ref.cast %[[X_REF_A_1]] : (!firrtl.rwprobe<uint<1>>) -> !firrtl.probe<uint>
    // CHECK-NEXT: %[[X_REF_B:.+]] = firrtl.ref.sub %[[X_REF]][1]
    // CHECK-NEXT: %[[X_REF_B_CAST:.+]] = firrtl.ref.cast %[[X_REF_B]] : (!firrtl.rwprobe<uint<2>>) -> !firrtl.probe<uint<2>>
    // CHECK-NEXT: %[[X_CAST_A_0:.+]] = firrtl.ref.resolve %[[X_REF_A_0_CAST]]
    // CHECK-NEXT: %[[X_CAST_A_1:.+]] = firrtl.ref.resolve %[[X_REF_A_1_CAST]]
    // CHECK-NEXT: %[[X_CAST_B:.+]] = firrtl.ref.resolve %[[X_REF_B_CAST]]
    // CHECK-NEXT: firrtl.node %[[X_CAST_A_0]] : !firrtl.uint
    // CHECK-NEXT: firrtl.node %[[X_CAST_A_1]] : !firrtl.uint
    // CHECK-NEXT: firrtl.node %[[X_CAST_B]] : !firrtl.uint<2>
    %x_ref_cast = firrtl.ref.cast %x_ref : (!firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>) -> !firrtl.probe<bundle<a: vector<uint, 2>, b: uint<2>>>
    %x_ref_cast_resolve = firrtl.ref.resolve %x_ref_cast : !firrtl.probe<bundle<a: vector<uint, 2>, b: uint<2>>>
    %x_ref_cast_node = firrtl.node %x_ref_cast_resolve : !firrtl.bundle<a: vector<uint, 2>, b: uint<2>>

    // Check resolve of rwprobe is preserved.
    // CHECK-NEXT: = firrtl.ref.resolve %[[X_REF]]
    // CHECK: firrtl.matchingconnect %bov_a_0,
    // CHECK: firrtl.matchingconnect %bov_a_1,
    // CHECK: firrtl.matchingconnect %bov_b,
    %x_read = firrtl.ref.resolve %x_ref : !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>
    firrtl.matchingconnect %bov, %x_read : !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>
    // CHECK-NEXT: }
  }
  // Check how rwprobe's of aggregates in instances are handled.
  // COMMON-LABEL: firrtl.module private @InstWithRWProbeOfAgg
  firrtl.module private @InstWithRWProbeOfAgg(in %clock : !firrtl.clock, in %pred : !firrtl.uint<1>) {
    // CHECK: {{((%[^,]+, ){3})}}
    // CHECK-SAME: %[[BOV_REF:[^,]+]],
    // CHECK-SAME: %[[BOV_A_0:.+]],        %[[BOV_A_1:.+]],        %[[BOV_B:.+]],        %{{.+}} = firrtl.instance
    // CHECK-NOT: firrtl.probe
    // CHECK-SAME: probe: !firrtl.probe<uint<2>>)
    %inst_vec_ref, %inst_vec, %inst_bov_ref, %inst_bov, %inst_probe = firrtl.instance inst @RefTypeBV_RW(
      out vec_ref: !firrtl.rwprobe<vector<uint<1>, 2>>,
      out vec: !firrtl.vector<uint<1>, 2>,
      out bov_ref: !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>,
      out bov: !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>,
      out probe: !firrtl.probe<uint<2>>)

    // Check lowering force and release operations.
    // Use self-assigns for simplicity.
    // Source operand may need to be materialized from its elements.
    // CHECK: vectorcreate
    // CHECK: bundlecreate
    // CHECK: firrtl.ref.force %clock, %pred, %[[BOV_REF]],
    firrtl.ref.force %clock, %pred, %inst_bov_ref, %inst_bov : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>, !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>
    // CHECK: firrtl.ref.force_initial %pred, %[[BOV_REF]],
    firrtl.ref.force_initial %pred, %inst_bov_ref, %inst_bov : !firrtl.uint<1>, !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>, !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>
    // CHECK: firrtl.ref.release %clock, %pred, %[[BOV_REF]] :
    firrtl.ref.release %clock, %pred, %inst_bov_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>
    // CHECK: firrtl.ref.release_initial %pred, %[[BOV_REF]] :
    firrtl.ref.release_initial %pred, %inst_bov_ref : !firrtl.uint<1>, !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>
    // CHECK-NEXT: }
  }

  // COMMON-LABEL: firrtl.module private @ForeignTypes
  firrtl.module private @ForeignTypes() {
    // COMMON-NEXT: firrtl.wire : index
    %0 = firrtl.wire : index
  }

  // CHECK-LABEL: firrtl.module @MergeBundle
  firrtl.module @MergeBundle(out %o: !firrtl.bundle<valid: uint<1>, ready: uint<1>>, in %i: !firrtl.uint<1>)
  {
    %a = firrtl.wire   : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
    firrtl.matchingconnect %o, %a : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
    %0 = firrtl.bundlecreate %i, %i : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.bundle<valid: uint<1>, ready: uint<1>>
    firrtl.matchingconnect %a, %0 : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
    // CHECK:  %a_valid = firrtl.wire   : !firrtl.uint<1>
    // CHECK:  %a_ready = firrtl.wire   : !firrtl.uint<1>
    // CHECK:  firrtl.matchingconnect %o_valid, %a_valid : !firrtl.uint<1>
    // CHECK:  firrtl.matchingconnect %o_ready, %a_ready : !firrtl.uint<1>
    // CHECK:  firrtl.matchingconnect %a_valid, %i : !firrtl.uint<1>
    // CHECK:  firrtl.matchingconnect %a_ready, %i : !firrtl.uint<1>
    // AGGREGATE: firrtl.bundlecreate
    // SIG: firrtl.bundlecreate
  }

  // COMMON-LABEL: firrtl.module @MergeVector
  firrtl.module @MergeVector(out %o: !firrtl.vector<uint<1>, 3>, in %i: !firrtl.uint<1>) {
    %a = firrtl.wire   : !firrtl.vector<uint<1>, 3>
    firrtl.matchingconnect %o, %a : !firrtl.vector<uint<1>, 3>
    %0 = firrtl.vectorcreate %i, %i, %i : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.vector<uint<1>, 3>
    firrtl.matchingconnect %a, %0 : !firrtl.vector<uint<1>, 3>
    // CHECK:  %a_0 = firrtl.wire   : !firrtl.uint<1>
    // CHECK:  %a_1 = firrtl.wire   : !firrtl.uint<1>
    // CHECK:  %a_2 = firrtl.wire   : !firrtl.uint<1>
    // CHECK:  firrtl.matchingconnect %o_0, %a_0 : !firrtl.uint<1>
    // CHECK:  firrtl.matchingconnect %o_1, %a_1 : !firrtl.uint<1>
    // CHECK:  firrtl.matchingconnect %o_2, %a_2 : !firrtl.uint<1>
    // CHECK:  firrtl.matchingconnect %a_0, %i : !firrtl.uint<1>
    // CHECK:  firrtl.matchingconnect %a_1, %i : !firrtl.uint<1>
    // CHECK:  firrtl.matchingconnect %a_2, %i : !firrtl.uint<1>
    // AGGREGATE: firrtl.vectorcreate
    // SIG: firrtl.vectorcreate
  }

  // Check that an instance with attributes known and unknown to FIRRTL Dialect
  // are copied to the lowered instance.
  firrtl.module @SubmoduleWithAggregate(out %a: !firrtl.vector<uint<1>, 1>) {}
  // COMMON-LABEL: firrtl.module @ModuleWithInstanceAttributes
  firrtl.module @ModuleWithInstanceAttributes() {
    // COMMON: firrtl.instance
    // COMMON-SAME:   hello = "world"
    // COMMON-SAME:   lowerToBind
    // COMMON-SAME:   output_file = #hw.output_file<"Foo.sv">
    %sub_a = firrtl.instance sub {
      hello = "world",
      lowerToBind,
      output_file = #hw.output_file<"Foo.sv">
    } @SubmoduleWithAggregate(out a: !firrtl.vector<uint<1>, 1>)
  }

  // COMMON-LABEL: firrtl.module @ElementWise
  firrtl.module @ElementWise(in %a: !firrtl.vector<uint<1>, 1>, in %b: !firrtl.vector<uint<1>, 1>, out %c_0: !firrtl.vector<uint<1>, 1>, out %c_1: !firrtl.vector<uint<1>, 1>, out %c_2: !firrtl.vector<uint<1>, 1>) {
    // CHECK-NEXT: %0 = firrtl.or %a_0, %b_0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: firrtl.matchingconnect %c_0_0, %0 : !firrtl.uint<1>
    // CHECK-NEXT: %1 = firrtl.and %a_0, %b_0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: firrtl.matchingconnect %c_1_0, %1 : !firrtl.uint<1>
    // CHECK-NEXT: %2 = firrtl.xor %a_0, %b_0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: firrtl.matchingconnect %c_2_0, %2 : !firrtl.uint<1>
    // Check that elementwise_* are preserved.
    // AGGREGATE: firrtl.elementwise_or
    // AGGREGATE: firrtl.elementwise_and
    // AGGREGATE: firrtl.elementwise_xor
    // SIG: firrtl.elementwise_or
    // SIG: firrtl.elementwise_and
    // SIG: firrtl.elementwise_xor
    %0 = firrtl.elementwise_or %a, %b : (!firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>) -> !firrtl.vector<uint<1>, 1>
    firrtl.matchingconnect %c_0, %0 : !firrtl.vector<uint<1>, 1>
    %1 = firrtl.elementwise_and %a, %b : (!firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>) -> !firrtl.vector<uint<1>, 1>
    firrtl.matchingconnect %c_1, %1 : !firrtl.vector<uint<1>, 1>
    %2 = firrtl.elementwise_xor %a, %b : (!firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>) -> !firrtl.vector<uint<1>, 1>
    firrtl.matchingconnect %c_2, %2 : !firrtl.vector<uint<1>, 1>
  }

  // COMMON-LABEL: firrtl.module @MuxInt
  firrtl.module @MuxInt(in %sel1: !firrtl.uint<1>, in %sel2: !firrtl.uint<2>, in %v1: !firrtl.bundle<a: uint<5>>, in %v0: !firrtl.bundle<a: uint<5>>, out %out1: !firrtl.bundle<a: uint<5>>, out %out2: !firrtl.bundle<a: uint<5>>) {
    // CHECK: firrtl.int.mux4cell(%sel2, %v1_a, %v0_a, %v1_a, %v0_a) : (!firrtl.uint<2>, !firrtl.uint<5>, !firrtl.uint<5>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
    %0 = firrtl.int.mux4cell(%sel2, %v1, %v0, %v1, %v0) : (!firrtl.uint<2>, !firrtl.bundle<a: uint<5>>, !firrtl.bundle<a: uint<5>>, !firrtl.bundle<a: uint<5>>, !firrtl.bundle<a: uint<5>>) -> !firrtl.bundle<a: uint<5>>
    firrtl.matchingconnect %out1, %0 : !firrtl.bundle<a: uint<5>>
    // CHECK: firrtl.int.mux2cell(%sel1, %v1_a, %v0_a) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
    %1 = firrtl.int.mux2cell(%sel1, %v1, %v0) : (!firrtl.uint<1>, !firrtl.bundle<a: uint<5>>, !firrtl.bundle<a: uint<5>>) -> !firrtl.bundle<a: uint<5>>
    firrtl.matchingconnect %out2, %0 : !firrtl.bundle<a: uint<5>>
    // SIG: firrtl.int.mux4cell(%sel2, %v1, %v0, %v1, %v0) : (!firrtl.uint<2>, !firrtl.bundle<a: uint<5>>, !firrtl.bundle<a: uint<5>>, !firrtl.bundle<a: uint<5>>, !firrtl.bundle<a: uint<5>>) -> !firrtl.bundle<a: uint<5>>
    // SIG: firrtl.int.mux2cell(%sel1, %v1, %v0) : (!firrtl.uint<1>, !firrtl.bundle<a: uint<5>>, !firrtl.bundle<a: uint<5>>) -> !firrtl.bundle<a: uint<5>>
  }

  // COMMON-LABEL: firrtl.module @Groups
  firrtl.layer @GroupFoo bind {}
  firrtl.module @Groups() {
    // COMMON-NEXT: firrtl.layerblock @GroupFoo
    firrtl.layerblock @GroupFoo {
      // CHECK-NEXT: %a_b = firrtl.wire : !firrtl.uint<1>
      // SIG-NEXT: %a = firrtl.wire : !firrtl.bundle<b: uint<1>>
      %a = firrtl.wire : !firrtl.bundle<b: uint<1>>
    }
  }

  // CHECK-LABEL:  firrtl.module @Alias
  // CHECK-NOT: alias
  // AGGREGATE-LABEL: firrtl.module @Alias
  // AGGREGATE-SAME: alias<FooBundle, bundle<x: uint<32>, y: uint<32>>>
  // SIG-LABEL: firrtl.module @Alias
  // SIG-SAME: alias<FooBundle, bundle<x: uint<32>, y: uint<32>>>
  firrtl.module @Alias(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %io: !firrtl.bundle<in flip: alias<FooBundle, bundle<x: uint<32>, y: uint<32>>>, out: alias<FooBundle, bundle<x: uint<32>, y: uint<32>>>>) {
  }
} // CIRCUIT

// Check that we don't lose the DontTouchAnnotation when it is not the last
// annotation in the list of annotations.
// https://github.com/llvm/circt/issues/3504
// CHECK-LABEL: firrtl.circuit "DontTouch"
firrtl.circuit "DontTouch" {
  // CHECK: in %port_field: !firrtl.uint<1> [{class = "firrtl.transforms.DontTouchAnnotation"}, {class = "Test"}]
  // SIG:  in %port: !firrtl.bundle<field: uint<1>> [{circt.fieldID = 1 : i32, class = "firrtl.transforms.DontTouchAnnotation"}, {circt.fieldID = 1 : i32, class = "Test"}]
  firrtl.module @DontTouch (in %port: !firrtl.bundle<field: uint<1>> [
    {circt.fieldID = 1 : i32, class = "firrtl.transforms.DontTouchAnnotation"},
    {circt.fieldID = 1 : i32, class = "Test"}
  ]) {
 }
}

// Check that we don't create symbols for non-local annotations.
firrtl.circuit "Foo"  {
  hw.hierpath private @nla [@Foo::@bar, @Bar]
  // CHECK:       firrtl.module private @Bar(in %a_b:
  // CHECK-SAME:    !firrtl.uint<1> [{circt.nonlocal = @nla, class = "circt.test"}])
  firrtl.module private @Bar(in %a: !firrtl.bundle<b: uint<1>>
      [{circt.fieldID = 1 : i32, circt.nonlocal = @nla, class = "circt.test"}]) {
  }
  firrtl.module @Foo() {
    %bar_a = firrtl.instance bar sym @bar @Bar(in a: !firrtl.bundle<b: uint<1>>)
    %invalid = firrtl.invalidvalue : !firrtl.bundle<b: uint<1>>
    firrtl.matchingconnect %bar_a, %invalid : !firrtl.bundle<b: uint<1>>
  }
}

// Check handling of inner symbols.
// COMMON-LABEL: circuit "InnerSym"
firrtl.circuit "InnerSym" {
  // COMMON-LABEL: module @InnerSym(
  // CHECK-SAME:  in %x_a: !firrtl.uint<5>, in %x_b: !firrtl.uint<3> sym @x)
  // AGGREGATE-SAME: in %x: !firrtl.bundle<a: uint<5>, b: uint<3>> sym [<@x,2,public>])
  // SIG-SAME: in %x: !firrtl.bundle<a: uint<5>, b: uint<3>> sym [<@x,2,public>])
  firrtl.module @InnerSym(in %x: !firrtl.bundle<a: uint<5>, b: uint<3>> sym [<@x,2,public>]) { }

  // COMMON-LABEL: module @InnerSymMore(
  // CHECK-SAME: in %x_a_x_1: !firrtl.uint<3> sym @x_1
  // CHECK-SAME: in %x_a_y: !firrtl.uint<2> sym @y
  firrtl.module @InnerSymMore(in %x: !firrtl.bundle<a: bundle<x: vector<uint<3>, 2>, y: uint<2>>, b: uint<3>> sym [<@y,5, public>,<@x_1,4,public>]) { }
}

// Check handling of wire of probe-aggregate.
// COMMON-LABEL: "WireProbe"
firrtl.circuit "WireProbe" {
  // (Read-only probes are unconditionally split as workaround for 4479)
  // COMMON-LABEL: module public @WireProbe
  // LT-SAME: out %out_a: !firrtl.probe<uint<5>>
  // LT-SAME: out %out_b: !firrtl.probe<uint<5>>
  firrtl.module public @WireProbe(out %out: !firrtl.probe<bundle<a: uint<5>, b: uint<5>>>,
                                  out %y: !firrtl.probe<uint<5>>,
                                  out %z: !firrtl.probe<uint<5>>) {
    // LT-NEXT: %w_a = firrtl.wire
    // LT-NEXT: %w_b = firrtl.wire
    // LT-NOT: firrtl.ref.sub
    // COMMON: }
    %w = firrtl.wire : !firrtl.probe<bundle<a: uint<5>, b: uint<5>>>
    %0 = firrtl.ref.sub %w[1] : !firrtl.probe<bundle<a: uint<5>, b: uint<5>>>
    %1 = firrtl.ref.sub %w[0] : !firrtl.probe<bundle<a: uint<5>, b: uint<5>>>
    firrtl.ref.define %y, %1 : !firrtl.probe<uint<5>>
    firrtl.ref.define %z, %0 : !firrtl.probe<uint<5>>
    firrtl.ref.define %out, %w : !firrtl.probe<bundle<a: uint<5>, b: uint<5>>>
  }
}

firrtl.circuit "UnrealizedConversion" {
  // COMMON-LABEL: @UnrealizedConversion
  firrtl.module @UnrealizedConversion( ){
    // CHECK: %[[a:.+]] = "d.w"() : () -> !hw.struct<data: i64, tag: i1>
    // CHECK: = builtin.unrealized_conversion_cast %[[a]] : !hw.struct<data: i64, tag: i1> to !firrtl.bundle<data: uint<64>, tag: uint<1>>
    %a = "d.w"() : () -> (!hw.struct<data: i64, tag: i1>)
    %b = builtin.unrealized_conversion_cast %a : !hw.struct<data: i64, tag: i1> to !firrtl.bundle<data: uint<64>, tag: uint<1>>
    %w = firrtl.wire : !firrtl.bundle<data: uint<64>, tag: uint<1>>
    firrtl.matchingconnect %w, %b : !firrtl.bundle<data: uint<64>, tag: uint<1>>
  }
}

firrtl.circuit "Conventions1" {
  // COMMON-LABEL: @Conventions1
  // AGGREGATE-SAME: %input_0
  // AGGREGATE-NEXT: firrtl.reg
  // AGGREGATE-SAME: !firrtl.vector<uint<8>, 1>
  firrtl.module public @Conventions1(in %input: !firrtl.vector<uint<8>, 1>, in %clk: !firrtl.clock, out %port: !firrtl.vector<uint<8>, 1>) attributes {convention = #firrtl<convention scalarized>, body_type_lowering = #firrtl<convention internal>}{
    %r = firrtl.reg interesting_name %clk : !firrtl.clock, !firrtl.vector<uint<8>, 1>
    firrtl.matchingconnect %r, %input : !firrtl.vector<uint<8>, 1>
    firrtl.matchingconnect %port, %r : !firrtl.vector<uint<8>, 1>
  }
  // COMMON-LABEL: @Conventions2
  // AGGREGATE-SAME: %input_0: !firrtl.uint<8>
  // AGGREGATE-NEXT: firrtl.reg
  // AGGREGATE-SAME: !firrtl.uint<8>
  firrtl.module private @Conventions2(in %input: !firrtl.vector<uint<8>, 1>, in %clk: !firrtl.clock, out %port: !firrtl.vector<uint<8>, 1>) attributes {convention = #firrtl<convention scalarized>, body_type_lowering = #firrtl<convention scalarized>}{
    %r = firrtl.reg interesting_name %clk : !firrtl.clock, !firrtl.vector<uint<8>, 1>
    firrtl.matchingconnect %r, %input : !firrtl.vector<uint<8>, 1>
    firrtl.matchingconnect %port, %r : !firrtl.vector<uint<8>, 1>
  }
  // COMMON-LABEL: @Conventions3
  // AGGREGATE-SAME: %input: !firrtl.vector<uint<8>, 1>
  // AGGREGATE-NEXT: firrtl.reg
  // AGGREGATE-SAME: !firrtl.vector<uint<8>, 1>
  firrtl.module private @Conventions3(in %input: !firrtl.vector<uint<8>, 1>, in %clk: !firrtl.clock, out %port: !firrtl.vector<uint<8>, 1>) attributes {convention = #firrtl<convention internal>, body_type_lowering = #firrtl<convention internal>}{
    %r = firrtl.reg interesting_name %clk : !firrtl.clock, !firrtl.vector<uint<8>, 1>
    firrtl.matchingconnect %r, %input : !firrtl.vector<uint<8>, 1>
    firrtl.matchingconnect %port, %r : !firrtl.vector<uint<8>, 1>
  }
  // COMMON-LABEL: @Conventions4
  // AGGREGATE-SAME: %input: !firrtl.vector<uint<8>, 1>
  // AGGREGATE-NEXT: firrtl.reg
  // AGGREGATE-SAME: !firrtl.uint<8>
  firrtl.module private @Conventions4(in %input: !firrtl.vector<uint<8>, 1>, in %clk: !firrtl.clock, out %port: !firrtl.vector<uint<8>, 1>) attributes {convention = #firrtl<convention internal>, body_type_lowering = #firrtl<convention scalarized>}{
    %r = firrtl.reg interesting_name %clk : !firrtl.clock, !firrtl.vector<uint<8>, 1>
    firrtl.matchingconnect %r, %input : !firrtl.vector<uint<8>, 1>
    firrtl.matchingconnect %port, %r : !firrtl.vector<uint<8>, 1>
  }
}

// Test that memories have their prefixes copied when lowering.
// See: https://github.com/llvm/circt/issues/7835
firrtl.circuit "MemoryPrefixCopying" {
  // COMMON-LABEL: firrtl.module @MemoryPrefixCopying
  firrtl.module @MemoryPrefixCopying() {
    // COMMON:      firrtl.mem
    // COMMON-SAME:   prefix = "Foo_"
    %mem_r = firrtl.mem Undefined {
      depth = 64 : i64,
      name = "ram",
      portNames = ["r"],
      prefix = "Foo_",
      readLatency = 1 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<6>, en: uint<1>, clk: clock, data flip: vector<uint<8>, 1>>
  }
}


firrtl.circuit "DiscardableAttributes" {
  // COMMON-LABEL: firrtl.module @DiscardableAttributes
  firrtl.module @DiscardableAttributes(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a: !firrtl.bundle<a: uint<1>>) {
    // CHECK-NEXT: %node_a = firrtl.node %a_a {foo}
    // CHECK-NEXT: %wire_a = firrtl.wire {foo = "bar"}
    // CHECK-NEXT: %reg_a = firrtl.reg %clock {foo}
    // CHECK-NEXT: %regreset_a = firrtl.regreset %clock, %reset, %a_a {foo}
    %node = firrtl.node %a {foo}: !firrtl.bundle<a: uint<1>>
    %wire = firrtl.wire {foo = "bar"} : !firrtl.bundle<a: uint<1>>
    %reg = firrtl.reg %clock {foo}: !firrtl.clock, !firrtl.bundle<a: uint<1>>
    %regreset = firrtl.regreset %clock, %reset, %a {foo}: !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
  }
}
