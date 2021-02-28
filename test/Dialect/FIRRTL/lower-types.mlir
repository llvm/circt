// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl.module(firrtl-lower-types))' -split-input-file %s | FileCheck %s

firrtl.circuit "TopLevel" {

  // CHECK-LABEL: firrtl.module @Simple
  // CHECK-SAME: %[[SOURCE_VALID_NAME:source_valid]]: [[SOURCE_VALID_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: %[[SOURCE_READY_NAME:source_ready]]: [[SOURCE_READY_TYPE:!firrtl.flip<uint<1>>]]
  // CHECK-SAME: %[[SOURCE_DATA_NAME:source_data]]: [[SOURCE_DATA_TYPE:!firrtl.uint<64>]]
  // CHECK-SAME: %[[SINK_VALID_NAME:sink_valid]]: [[SINK_VALID_TYPE:!firrtl.flip<uint<1>>]]
  // CHECK-SAME: %[[SINK_READY_NAME:sink_ready]]: [[SINK_READY_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: %[[SINK_DATA_NAME:sink_data]]: [[SINK_DATA_TYPE:!firrtl.flip<uint<64>>]]
  firrtl.module @Simple(%source: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>,
                        %sink: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) {

    // CHECK-NEXT: firrtl.when %[[SOURCE_VALID_NAME]]
    // CHECK-NEXT:   firrtl.connect %[[SINK_DATA_NAME]], %[[SOURCE_DATA_NAME]] : [[SINK_DATA_TYPE]], [[SOURCE_DATA_TYPE]]
    // CHECK-NEXT:   firrtl.connect %[[SINK_VALID_NAME]], %[[SOURCE_VALID_NAME]] : [[SINK_VALID_TYPE]], [[SOURCE_VALID_TYPE]]
    // CHECK-NEXT:   firrtl.connect %[[SOURCE_READY_NAME]], %[[SINK_READY_NAME]] : [[SOURCE_READY_TYPE]], [[SINK_READY_TYPE]]

    %0 = firrtl.subfield %source("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %source("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.flip<uint<1>>
    %2 = firrtl.subfield %source("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
    %3 = firrtl.subfield %sink("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<1>>
    %4 = firrtl.subfield %sink("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.uint<1>
    %5 = firrtl.subfield %sink("data") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<64>>
    firrtl.when %0 {
      firrtl.connect %5, %2 : !firrtl.flip<uint<64>>, !firrtl.uint<64>
      firrtl.connect %3, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
      firrtl.connect %1, %4 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    }
  }

  // CHECK-LABEL: firrtl.module @TopLevel
  // CHECK-SAME: %source_valid: [[SOURCE_VALID_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: %source_ready: [[SOURCE_READY_TYPE:!firrtl.flip<uint<1>>]]
  // CHECK-SAME: %source_data: [[SOURCE_DATA_TYPE:!firrtl.uint<64>]]
  // CHECK-SAME: %sink_valid: [[SINK_VALID_TYPE:!firrtl.flip<uint<1>>]]
  // CHECK-SAME: %sink_ready: [[SINK_READY_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: %sink_data: [[SINK_DATA_TYPE:!firrtl.flip<uint<64>>]]
  firrtl.module @TopLevel(%source: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>,
                          %sink: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) {

    // CHECK-NEXT: %inst_source_valid, %inst_source_ready, %inst_source_data, %inst_sink_valid, %inst_sink_ready, %inst_sink_data
    // CHECK-SAME: = firrtl.instance @Simple {name = "", portNames = ["source_valid", "source_ready", "source_data", "sink_valid", "sink_ready", "sink_data"]} :
    // CHECK-SAME: !firrtl.flip<uint<1>>, !firrtl.uint<1>, !firrtl.flip<uint<64>>, !firrtl.uint<1>, !firrtl.flip<uint<1>>, !firrtl.uint<64>
    %sourceV, %sinkV = firrtl.instance @Simple {name = "", portNames = ["source", "sink"]} : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>

    // CHECK-NEXT: firrtl.connect %inst_source_valid, %source_valid
    // CHECK-NEXT: firrtl.connect %source_ready, %inst_source_ready
    // CHECK-NEXT: firrtl.connect %inst_source_data, %source_data
    // CHECK-NEXT: firrtl.connect %sink_valid, %inst_sink_valid
    // CHECK-NEXT: firrtl.connect %inst_sink_ready, %sink_ready
    // CHECK-NEXT: firrtl.connect %sink_data, %inst_sink_data
    firrtl.connect %sourceV, %source : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>

    firrtl.connect %sink, %sinkV : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
  }
}

// -----

firrtl.circuit "Recursive" {

  // CHECK-LABEL: firrtl.module @Recursive
  // CHECK-SAME: %[[FLAT_ARG_1_NAME:arg_foo_bar_baz]]: [[FLAT_ARG_1_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: %[[FLAT_ARG_2_NAME:arg_foo_qux]]: [[FLAT_ARG_2_TYPE:!firrtl.sint<64>]]
  // CHECK-SAME: %[[OUT_1_NAME:out1]]: [[OUT_1_TYPE:!firrtl.flip<uint<1>>]]
  // CHECK-SAME: %[[OUT_2_NAME:out2]]: [[OUT_2_TYPE:!firrtl.flip<sint<64>>]]
  firrtl.module @Recursive(%arg: !firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>,
                           %out1: !firrtl.flip<uint<1>>, %out2: !firrtl.flip<sint<64>>) {

    // CHECK-NEXT: firrtl.connect %[[OUT_1_NAME]], %[[FLAT_ARG_1_NAME]] : [[OUT_1_TYPE]], [[FLAT_ARG_1_TYPE]]
    // CHECK-NEXT: firrtl.connect %[[OUT_2_NAME]], %[[FLAT_ARG_2_NAME]] : [[OUT_2_TYPE]], [[FLAT_ARG_2_TYPE]]

    %0 = firrtl.subfield %arg("foo") : (!firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>) -> !firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>
    %1 = firrtl.subfield %0("bar") : (!firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>) -> !firrtl.bundle<baz: uint<1>>
    %2 = firrtl.subfield %1("baz") : (!firrtl.bundle<baz: uint<1>>) -> !firrtl.uint<1>
    %3 = firrtl.subfield %0("qux") : (!firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>) -> !firrtl.sint<64>
    firrtl.connect %out1, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    firrtl.connect %out2, %3 : !firrtl.flip<sint<64>>, !firrtl.sint<64>
  }

}

// -----

firrtl.circuit "Uniquification" {

  // CHECK-LABEL: firrtl.module @Uniquification
  // CHECK-SAME: %[[FLATTENED_ARG:a_b]]: [[FLATTENED_TYPE:!firrtl.uint<1>]],
  // CHECK-NOT: %[[FLATTENED_ARG]]
  // CHECK-SAME: %[[RENAMED_ARG:a_b.+]]: [[RENAMED_TYPE:!firrtl.uint<1>]] {firrtl.name = "[[FLATTENED_ARG]]"}
  firrtl.module @Uniquification(%a: !firrtl.bundle<b: uint<1>>, %a_b: !firrtl.uint<1>) {
  }

}

// -----

firrtl.circuit "Top" {

  // CHECK-LABEL: firrtl.module @Top
  firrtl.module @Top(%in : !firrtl.bundle<a: uint<1>, b: uint<1>>,
                     %out : !firrtl.flip<bundle<a: uint<1>, b: uint<1>>>) {
    // CHECK: firrtl.connect %out_a, %in_a : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    // CHECK: firrtl.connect %out_b, %in_b : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    firrtl.connect %out, %in : !firrtl.flip<bundle<a: uint<1>, b: uint<1>>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
  }

}

// -----

firrtl.circuit "Foo" {
  // CHECK-LABEL: firrtl.module @Foo
  // CHECK-SAME: %[[FLAT_ARG_INPUT_NAME:a_b_c]]: [[FLAT_ARG_INPUT_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: %[[FLAT_ARG_OUTPUT_NAME:b_b_c]]: [[FLAT_ARG_OUTPUT_TYPE:!firrtl.flip<uint<1>>]]
  firrtl.module @Foo(%a: !firrtl.bundle<b: bundle<c: uint<1>>>, %b: !firrtl.flip<bundle<b: bundle<c: uint<1>>>>) {
    // CHECK: firrtl.connect %[[FLAT_ARG_OUTPUT_NAME]], %[[FLAT_ARG_INPUT_NAME]] : [[FLAT_ARG_OUTPUT_TYPE]], [[FLAT_ARG_INPUT_TYPE]]
    firrtl.connect %b, %a : !firrtl.flip<bundle<b: bundle<c: uint<1>>>>, !firrtl.bundle<b: bundle<c: uint<1>>>
  }
}

// -----

// COM: Test lower of a 1-read 1-write aggregate memory
//
// COM: circuit Foo :
// COM:   module Foo :
// COM:     input clock: Clock
// COM:     input rAddr: UInt<4>
// COM:     input rEn: UInt<1>
// COM:     output rData: {a: UInt<8>, b: UInt<8>}
// COM:     input wAddr: UInt<4>
// COM:     input wEn: UInt<1>
// COM:     input wMask: {a: UInt<1>, b: UInt<1>}
// COM:     input wData: {a: UInt<8>, b: UInt<8>}
// COM:
// COM:     mem memory:
// COM:       data-type => {a: UInt<8>, b: UInt<8>}
// COM:       depth => 16
// COM:       reader => r
// COM:       writer => w
// COM:       read-latency => 0
// COM:       write-latency => 1
// COM:       read-under-write => undefined
// COM:
// COM:     memory.r.clk <= clock
// COM:     memory.r.en <= rEn
// COM:     memory.r.addr <= rAddr
// COM:     rData <= memory.r.data
// COM:
// COM:     memory.w.clk <= clock
// COM:     memory.w.en <= wEn
// COM:     memory.w.addr <= wAddr
// COM:     memory.w.mask <= wMask
// COM:     memory.w.data <= wData

firrtl.circuit "Foo" {

  // CHECK-LABEL: firrtl.module @Foo
  firrtl.module @Foo(%clock: !firrtl.clock, %rAddr: !firrtl.uint<4>, %rEn: !firrtl.uint<1>, %rData: !firrtl.flip<bundle<a: uint<8>, b: uint<8>>>, %wAddr: !firrtl.uint<4>, %wEn: !firrtl.uint<1>, %wMask: !firrtl.bundle<a: uint<1>, b: uint<1>>, %wData: !firrtl.bundle<a: uint<8>, b: uint<8>>) {
    %memory_r, %memory_w = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r", "w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: bundle<a: uint<8>, b: uint<8>>>, !firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>>
    %0 = firrtl.subfield %memory_r("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: bundle<a: uint<8>, b: uint<8>>>) -> !firrtl.flip<clock>
    firrtl.connect %0, %clock : !firrtl.flip<clock>, !firrtl.clock
    %1 = firrtl.subfield %memory_r("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: bundle<a: uint<8>, b: uint<8>>>) -> !firrtl.flip<uint<1>>
    firrtl.connect %1, %rEn : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    %2 = firrtl.subfield %memory_r("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: bundle<a: uint<8>, b: uint<8>>>) -> !firrtl.flip<uint<4>>
    firrtl.connect %2, %rAddr : !firrtl.flip<uint<4>>, !firrtl.uint<4>
    %3 = firrtl.subfield %memory_r("data") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: bundle<a: uint<8>, b: uint<8>>>) -> !firrtl.bundle<a: uint<8>, b: uint<8>>
    firrtl.connect %rData, %3 : !firrtl.flip<bundle<a: uint<8>, b: uint<8>>>, !firrtl.bundle<a: uint<8>, b: uint<8>>
    %4 = firrtl.subfield %memory_w("clk") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>>) -> !firrtl.flip<clock>
    firrtl.connect %4, %clock : !firrtl.flip<clock>, !firrtl.clock
    %5 = firrtl.subfield %memory_w("en") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>>) -> !firrtl.flip<uint<1>>
    firrtl.connect %5, %wEn : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    %6 = firrtl.subfield %memory_w("addr") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>>) -> !firrtl.flip<uint<4>>
    firrtl.connect %6, %wAddr : !firrtl.flip<uint<4>>, !firrtl.uint<4>
    %7 = firrtl.subfield %memory_w("mask") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>>) -> !firrtl.flip<bundle<a: uint<1>, b: uint<1>>>
    firrtl.connect %7, %wMask : !firrtl.flip<bundle<a: uint<1>, b: uint<1>>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
    %8 = firrtl.subfield %memory_w("data") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>>) -> !firrtl.flip<bundle<a: uint<8>, b: uint<8>>>
    firrtl.connect %8, %wData : !firrtl.flip<bundle<a: uint<8>, b: uint<8>>>, !firrtl.bundle<a: uint<8>, b: uint<8>>

    // COM: ---------------------------------------------------------------------------------
    // COM: Split memory "a" should exist
    // CHECK: %[[MEMORY_A_R:.+]], %[[MEMORY_A_W:.+]] = firrtl.mem {{.+}} data: uint<8>, mask: uint<1>
    // COM: ---------------------------------------------------------------------------------
    // COM: Read port
    // CHECK-DAG: %[[MEMORY_A_R_ADDR:.+]] = firrtl.subfield %[[MEMORY_A_R]]("addr")
    // CHECK-DAG: %[[MEMORY_R_ADDR:.+]] = firrtl.wire
    // CHECK: firrtl.connect %[[MEMORY_A_R_ADDR]], %[[MEMORY_R_ADDR]]
    // CHECK-DAG: %[[MEMORY_A_R_EN:.+]] = firrtl.subfield %[[MEMORY_A_R]]("en")
    // CHECK-DAG: %[[MEMORY_R_EN:.+]] = firrtl.wire
    // CHECK: firrtl.connect %[[MEMORY_A_R_EN]], %[[MEMORY_R_EN]]
    // CHECK-DAG: %[[MEMORY_A_R_CLK:.+]] = firrtl.subfield %[[MEMORY_A_R]]("clk")
    // CHECK-DAG: %[[MEMORY_R_CLK:.+]] = firrtl.wire
    // CHECK: firrtl.connect %[[MEMORY_A_R_CLK]], %[[MEMORY_R_CLK]]
    // CHECK: %[[MEMORY_A_R_DATA:.+]] = firrtl.subfield %[[MEMORY_A_R]]("data")
    // COM: ---------------------------------------------------------------------------------
    // COM: Write Port
    // CHECK-DAG: %[[MEMORY_A_W_ADDR:.+]] = firrtl.subfield %[[MEMORY_A_W]]("addr")
    // CHECK-DAG: %[[MEMORY_W_ADDR:.+]] = firrtl.wire
    // CHECK: firrtl.connect %[[MEMORY_A_W_ADDR]], %[[MEMORY_W_ADDR]]
    // CHECK-DAG: %[[MEMORY_A_W_EN:.+]] = firrtl.subfield %[[MEMORY_A_W]]("en")
    // CHECK-DAG: %[[MEMORY_W_EN:.+]] = firrtl.wire
    // CHECK: firrtl.connect %[[MEMORY_A_W_EN]], %[[MEMORY_W_EN]]
    // CHECK-DAG: %[[MEMORY_A_W_CLK:.+]] = firrtl.subfield %[[MEMORY_A_W]]("clk")
    // CHECK-DAG: %[[MEMORY_W_CLK:.+]] = firrtl.wire
    // CHECK: firrtl.connect %[[MEMORY_A_W_CLK]], %[[MEMORY_W_CLK]]
    // CHECK: %[[MEMORY_A_W_DATA:.+]] = firrtl.subfield %[[MEMORY_A_W]]("data")
    // CHECK: %[[MEMORY_A_W_MASK:.+]] = firrtl.subfield %[[MEMORY_A_W]]("mask")
    // COM: ---------------------------------------------------------------------------------
    // COM: Split memory "b" should exist
    // CHECK: %[[MEMORY_B_R:.+]], %[[MEMORY_B_W:.+]] = firrtl.mem {{.+}} data: uint<8>, mask: uint<1>
    // COM: ---------------------------------------------------------------------------------
    // COM: Read port
    // CHECK: %[[MEMORY_B_R_ADDR:.+]] = firrtl.subfield %[[MEMORY_B_R]]("addr")
    // CHECK: firrtl.connect %[[MEMORY_B_R_ADDR]], %[[MEMORY_R_ADDR]]
    // CHECK: %[[MEMORY_B_R_EN:.+]] = firrtl.subfield %[[MEMORY_B_R]]("en")
    // CHECK: firrtl.connect %[[MEMORY_B_R_EN]], %[[MEMORY_R_EN]]
    // CHECK: %[[MEMORY_B_R_CLK:.+]] = firrtl.subfield %[[MEMORY_B_R]]("clk")
    // CHECK: firrtl.connect %[[MEMORY_B_R_CLK]], %[[MEMORY_R_CLK]]
    // CHECK: %[[MEMORY_B_R_DATA:.+]] = firrtl.subfield %[[MEMORY_B_R]]("data")
    // COM: ---------------------------------------------------------------------------------
    // COM: Write port
    // CHECK: %[[MEMORY_B_W_ADDR:.+]] = firrtl.subfield %[[MEMORY_B_W]]("addr")
    // CHECK: firrtl.connect %[[MEMORY_B_W_ADDR]], %[[MEMORY_W_ADDR]]
    // CHECK: %[[MEMORY_B_W_EN:.+]] = firrtl.subfield %[[MEMORY_B_W]]("en")
    // CHECK: firrtl.connect %[[MEMORY_B_W_EN]], %[[MEMORY_W_EN]]
    // CHECK: %[[MEMORY_B_W_CLK:.+]] = firrtl.subfield %[[MEMORY_B_W]]("clk")
    // CHECK: firrtl.connect %[[MEMORY_B_W_CLK]], %[[MEMORY_W_CLK]]
    // CHECK: %[[MEMORY_B_W_DATA:.+]] = firrtl.subfield %[[MEMORY_B_W]]("data")
    // CHECK: %[[MEMORY_B_W_MASK:.+]] = firrtl.subfield %[[MEMORY_B_W]]("mask")
    // COM: ---------------------------------------------------------------------------------
    // COM: Connections to module ports
    // CHECK: firrtl.connect %[[MEMORY_R_CLK]], %clock
    // CHECK: firrtl.connect %[[MEMORY_R_EN]], %rEn
    // CHECK: firrtl.connect %[[MEMORY_R_ADDR]], %rAddr
    // CHECK: firrtl.connect %rData_a, %[[MEMORY_A_R_DATA]]
    // CHECK: firrtl.connect %rData_b, %[[MEMORY_B_R_DATA]]
    // CHECK: firrtl.connect %[[MEMORY_W_CLK]], %clock
    // CHECK: firrtl.connect %[[MEMORY_W_EN]], %wEn
    // CHECK: firrtl.connect %[[MEMORY_W_ADDR]], %wAddr
    // CHECK: firrtl.connect %[[MEMORY_A_W_MASK]], %wMask_a
    // CHECK: firrtl.connect %[[MEMORY_B_W_MASK]], %wMask_b
    // CHECK: firrtl.connect %[[MEMORY_A_W_DATA]], %wData_a
    // CHECK: firrtl.connect %[[MEMORY_B_W_DATA]], %wData_b

  }
}


// -----
// https://github.com/llvm/circt/issues/593

module  {
  firrtl.circuit "top_mod" {
    firrtl.module @mod_2(%clock: !firrtl.clock, %inp_a: !firrtl.bundle<inp_d: uint<14>>) {
    }
    firrtl.module @top_mod(%clock: !firrtl.clock) {
      %U0_clock, %U0_inp_a = firrtl.instance @mod_2 {name = "U0", portNames = ["clock", "inp_a"]} : !firrtl.flip<clock>, !firrtl.flip<bundle<inp_d: uint<14>>>
      %0 = firrtl.invalidvalue : !firrtl.clock
      firrtl.connect %U0_clock, %0 : !firrtl.flip<clock>, !firrtl.clock
      %1 = firrtl.invalidvalue : !firrtl.bundle<inp_d: uint<14>>
      firrtl.connect %U0_inp_a, %1 : !firrtl.flip<bundle<inp_d: uint<14>>>, !firrtl.bundle<inp_d: uint<14>>
    }
  }
}

//CHECK-LABEL: module  {
//CHECK-NEXT:   firrtl.circuit "top_mod" {
//CHECK-NEXT:     firrtl.module @mod_2(%clock: !firrtl.clock, %inp_a_inp_d: !firrtl.uint<14>) {
//CHECK-NEXT:     }
//CHECK-NEXT:    firrtl.module @top_mod(%clock: !firrtl.clock) {
//CHECK-NEXT:      %U0_clock, %U0_inp_a_inp_d = firrtl.instance @mod_2 {name = "U0", portNames = ["clock", "inp_a_inp_d"]} : !firrtl.flip<clock>, !firrtl.flip<uint<14>>
//CHECK-NEXT:      %0 = firrtl.invalidvalue : !firrtl.clock
//CHECK-NEXT:      firrtl.connect %U0_clock, %0 : !firrtl.flip<clock>, !firrtl.clock
//CHECK-NEXT:      %1 = firrtl.invalidvalue : !firrtl.uint<14>
//CHECK-NEXT:      firrtl.connect %U0_inp_a_inp_d, %1 : !firrtl.flip<uint<14>>, !firrtl.uint<14>
//CHECK-NEXT:    }
//CHECK-NEXT:  }
//CHECK-NEXT:}

// -----
// https://github.com/llvm/circt/issues/661

// COM: This test is just checking that the following doesn't error.
module  {
  firrtl.circuit "foo" {
    firrtl.module @foo(%clock: !firrtl.clock) {
      %head_MPORT_2, %head_MPORT_6 = firrtl.mem Undefined {depth = 20 : i64, name = "head", portNames = ["MPORT_2", "MPORT_6"], readLatency = 0 : i32, writeLatency = 1 : i32}
      : !firrtl.flip<bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<5>, mask: uint<1>>>,
        !firrtl.flip<bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<5>, mask: uint<1>>>
      %127 = firrtl.subfield %head_MPORT_6("clk") : (!firrtl.flip<bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<5>, mask: uint<1>>>) -> !firrtl.flip<clock>
    }
  }
}

  firrtl.circuit "RegBundle" {
//CHECK-LABEL: firrtl.module @RegBundle(%a_a: !firrtl.uint<1>, %clk: !firrtl.clock, %b_a: !firrtl.flip<uint<1>>) {
//CHECK-NEXT: %x_a = firrtl.reg %clk {name = "x_a"} : (!firrtl.clock) -> !firrtl.uint<1>
//CHECK-NEXT: firrtl.connect %x_a, %a_a : !firrtl.uint<1>, !firrtl.uint<1>
//CHECK-NEXT: firrtl.connect %b_a, %x_a : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    firrtl.module @RegBundle(%a: !firrtl.bundle<a: uint<1>>, %clk: !firrtl.clock, %b: !firrtl.flip<bundle<a: uint<1>>>) {
      %x = firrtl.reg %clk {name = "x"} : (!firrtl.clock) -> !firrtl.bundle<a: uint<1>>
      %0 = firrtl.subfield %x("a") : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
      %1 = firrtl.subfield %a("a") : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %0, %1 : !firrtl.uint<1>, !firrtl.uint<1>
      %2 = firrtl.subfield %b("a") : (!firrtl.flip<bundle<a: uint<1>>>) -> !firrtl.flip<uint<1>>
      %3 = firrtl.subfield %x("a") : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %2, %3 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    }
}
  
// -----
// COM: Test vector lowering
firrtl.circuit "LowerVectors" {
  firrtl.module @LowerVectors(%a: !firrtl.vector<uint<1>, 2>, %b: !firrtl.flip<vector<uint<1>, 2>>) {
    firrtl.connect %b, %a: !firrtl.flip<vector<uint<1>, 2>>, !firrtl.vector<uint<1>, 2>
  }

  // CHECK: firrtl.module @LowerVectors(%a_0: !firrtl.uint<1>, %a_1: !firrtl.uint<1>, %b_0: !firrtl.flip<uint<1>>, %b_1: !firrtl.flip<uint<1>>)
  // CHECK: firrtl.connect %b_0, %a_0
  // CHECK: firrtl.connect %b_1, %a_1
}
