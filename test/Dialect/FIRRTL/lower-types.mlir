// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-lower-types)' %s | FileCheck --check-prefixes=CHECK,COMMON %s
// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-lower-types{flatten-mem=true})' %s | FileCheck --check-prefix=FLATTEN %s
// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-lower-types{preserve-aggregate=true})' %s | FileCheck --check-prefixes=AGGREGATE,COMMON %s


firrtl.circuit "TopLevel" {

  // COMMON-LABEL: firrtl.module @Simple
  // COMMON-SAME: in %[[SOURCE_VALID_NAME:source_valid]]: [[SOURCE_VALID_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: out %[[SOURCE_READY_NAME:source_ready]]: [[SOURCE_READY_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: in %[[SOURCE_DATA_NAME:source_data]]: [[SOURCE_DATA_TYPE:!firrtl.uint<64>]]
  // COMMON-SAME: out %[[SINK_VALID_NAME:sink_valid]]: [[SINK_VALID_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: in %[[SINK_READY_NAME:sink_ready]]: [[SINK_READY_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: out %[[SINK_DATA_NAME:sink_data]]: [[SINK_DATA_TYPE:!firrtl.uint<64>]]
  firrtl.module @Simple(in %source: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
                        out %sink: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {

    // COMMON-NEXT: firrtl.when %[[SOURCE_VALID_NAME]]
    // COMMON-NEXT:   firrtl.connect %[[SINK_DATA_NAME]], %[[SOURCE_DATA_NAME]] : [[SINK_DATA_TYPE]], [[SOURCE_DATA_TYPE]]
    // COMMON-NEXT:   firrtl.connect %[[SINK_VALID_NAME]], %[[SOURCE_VALID_NAME]] : [[SINK_VALID_TYPE]], [[SOURCE_VALID_TYPE]]
    // COMMON-NEXT:   firrtl.connect %[[SOURCE_READY_NAME]], %[[SINK_READY_NAME]] : [[SOURCE_READY_TYPE]], [[SINK_READY_TYPE]]

    %0 = firrtl.subfield %source(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %source(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
    %2 = firrtl.subfield %source(2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
    %3 = firrtl.subfield %sink(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
    %4 = firrtl.subfield %sink(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
    %5 = firrtl.subfield %sink(2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
    firrtl.when %0 {
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

    // COMMON-NEXT: %inst_source_valid, %inst_source_ready, %inst_source_data, %inst_sink_valid, %inst_sink_ready, %inst_sink_data
    // COMMON-SAME: = firrtl.instance "" @Simple(
    // COMMON-SAME: in source_valid: !firrtl.uint<1>, out source_ready: !firrtl.uint<1>, in source_data: !firrtl.uint<64>, out sink_valid: !firrtl.uint<1>, in sink_ready: !firrtl.uint<1>, out sink_data: !firrtl.uint<64>
    %sourceV, %sinkV = firrtl.instance "" @Simple(in source: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
                        out sink: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)

    // COMMON-NEXT: firrtl.connect %inst_source_valid, %source_valid
    // COMMON-NEXT: firrtl.connect %source_ready, %inst_source_ready
    // COMMON-NEXT: firrtl.connect %inst_source_data, %source_data
    // COMMON-NEXT: firrtl.connect %sink_valid, %inst_sink_valid
    // COMMON-NEXT: firrtl.connect %inst_sink_ready, %sink_ready
    // COMMON-NEXT: firrtl.connect %sink_data, %inst_sink_data
    firrtl.connect %sourceV, %source : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>

    firrtl.connect %sink, %sinkV : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
  }

  // COMMON-LABEL: firrtl.module @Recursive
  // CHECK-SAME: in %[[FLAT_ARG_1_NAME:arg_foo_bar_baz]]: [[FLAT_ARG_1_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: in %[[FLAT_ARG_2_NAME:arg_foo_qux]]: [[FLAT_ARG_2_TYPE:!firrtl.sint<64>]]
  // CHECK-SAME: out %[[OUT_1_NAME:out1]]: [[OUT_1_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: out %[[OUT_2_NAME:out2]]: [[OUT_2_TYPE:!firrtl.sint<64>]]
  // AGGREGATE-SAME: in %[[ARG_NAME:arg]]: [[ARG_TYPE:!firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>]]
  // AGGREGATE-SAME: out %[[OUT_1_NAME:out1]]: [[OUT_1_TYPE:!firrtl.uint<1>]]
  // AGGREGATE-SAME: out %[[OUT_2_NAME:out2]]: [[OUT_2_TYPE:!firrtl.sint<64>]]
  firrtl.module @Recursive(in %arg: !firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>,
                           out %out1: !firrtl.uint<1>, out %out2: !firrtl.sint<64>) {

    // CHECK-NEXT: firrtl.connect %[[OUT_1_NAME]], %[[FLAT_ARG_1_NAME]] : [[OUT_1_TYPE]], [[FLAT_ARG_1_TYPE]]
    // CHECK-NEXT: firrtl.connect %[[OUT_2_NAME]], %[[FLAT_ARG_2_NAME]] : [[OUT_2_TYPE]], [[FLAT_ARG_2_TYPE]]
    // AGGREGATE-NEXT:  %0 = firrtl.subfield %[[ARG_NAME]](0)
    // AGGREGATE-NEXT:  %1 = firrtl.subfield %0(0)
    // AGGREGATE-NEXT:  %2 = firrtl.subfield %1(0)
    // AGGREGATE-NEXT:  %3 = firrtl.subfield %0(1)
    // AGGREGATE-NEXT:  firrtl.connect %[[OUT_1_NAME]], %2
    // AGGREGATE-NEXT:  firrtl.connect %[[OUT_2_NAME]], %3

    %0 = firrtl.subfield %arg(0) : (!firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>) -> !firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>
    %1 = firrtl.subfield %0(0) : (!firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>) -> !firrtl.bundle<baz: uint<1>>
    %2 = firrtl.subfield %1(0) : (!firrtl.bundle<baz: uint<1>>) -> !firrtl.uint<1>
    %3 = firrtl.subfield %0(1) : (!firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>) -> !firrtl.sint<64>
    firrtl.connect %out1, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out2, %3 : !firrtl.sint<64>, !firrtl.sint<64>
  }

  // CHECK-LABEL: firrtl.module @Uniquification
  // CHECK-SAME: in %[[FLATTENED_ARG:a_b]]: [[FLATTENED_TYPE:!firrtl.uint<1>]],
  // CHECK-NOT: %[[FLATTENED_ARG]]
  // CHECK-SAME: in %[[RENAMED_ARG:a_b.+]]: [[RENAMED_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: {portNames = ["a_b", "a_b"]}
  firrtl.module @Uniquification(in %a: !firrtl.bundle<b: uint<1>>, in %a_b: !firrtl.uint<1>) {
  }

  // CHECK-LABEL: firrtl.module @Top
  firrtl.module @Top(in %in : !firrtl.bundle<a: uint<1>, b: uint<1>>,
                     out %out : !firrtl.bundle<a: uint<1>, b: uint<1>>) {
    // CHECK: firrtl.connect %out_a, %in_a : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.connect %out_b, %in_b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %in : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
  }

  // CHECK-LABEL: firrtl.module @Foo
  // CHECK-SAME: in %[[FLAT_ARG_INPUT_NAME:a_b_c]]: [[FLAT_ARG_INPUT_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: out %[[FLAT_ARG_OUTPUT_NAME:b_b_c]]: [[FLAT_ARG_OUTPUT_TYPE:!firrtl.uint<1>]]
  firrtl.module @Foo(in %a: !firrtl.bundle<b: bundle<c: uint<1>>>, out %b: !firrtl.bundle<b: bundle<c: uint<1>>>) {
    // CHECK: firrtl.connect %[[FLAT_ARG_OUTPUT_NAME]], %[[FLAT_ARG_INPUT_NAME]] : [[FLAT_ARG_OUTPUT_TYPE]], [[FLAT_ARG_INPUT_TYPE]]
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

  // CHECK-LABEL: firrtl.module @Mem2
  firrtl.module @Mem2(in %clock: !firrtl.clock, in %rAddr: !firrtl.uint<4>, in %rEn: !firrtl.uint<1>, out %rData: !firrtl.bundle<a: uint<8>, b: uint<8>>, in %wAddr: !firrtl.uint<4>, in %wEn: !firrtl.uint<1>, in %wMask: !firrtl.bundle<a: uint<1>, b: uint<1>>, in %wData: !firrtl.bundle<a: uint<8>, b: uint<8>>) {
    %memory_r, %memory_w = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r", "w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    %0 = firrtl.subfield %memory_r(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>) -> !firrtl.clock
    firrtl.connect %0, %clock : !firrtl.clock, !firrtl.clock
    %1 = firrtl.subfield %memory_r(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>) -> !firrtl.uint<1>
    firrtl.connect %1, %rEn : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.subfield %memory_r(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>) -> !firrtl.uint<4>
    firrtl.connect %2, %rAddr : !firrtl.uint<4>, !firrtl.uint<4>
    %3 = firrtl.subfield %memory_r(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>) -> !firrtl.bundle<a: uint<8>, b: uint<8>>
    firrtl.connect %rData, %3 : !firrtl.bundle<a: uint<8>, b: uint<8>>, !firrtl.bundle<a: uint<8>, b: uint<8>>
    %4 = firrtl.subfield %memory_w(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.clock
    firrtl.connect %4, %clock : !firrtl.clock, !firrtl.clock
    %5 = firrtl.subfield %memory_w(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<1>
    firrtl.connect %5, %wEn : !firrtl.uint<1>, !firrtl.uint<1>
    %6 = firrtl.subfield %memory_w(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<4>
    firrtl.connect %6, %wAddr : !firrtl.uint<4>, !firrtl.uint<4>
    %7 = firrtl.subfield %memory_w(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.connect %7, %wMask : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
    %8 = firrtl.subfield %memory_w(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<8>, b: uint<8>>
    firrtl.connect %8, %wData : !firrtl.bundle<a: uint<8>, b: uint<8>>, !firrtl.bundle<a: uint<8>, b: uint<8>>

    // ---------------------------------------------------------------------------------
    // Split memory "a" should exist
    // CHECK: %[[MEMORY_A_R:.+]], %[[MEMORY_A_W:.+]] = firrtl.mem {{.+}} data: uint<8>, mask: uint<1>
    //
    // Split memory "b" should exist
    // CHECK-NEXT: %[[MEMORY_B_R:.+]], %[[MEMORY_B_W:.+]] = firrtl.mem {{.+}} data: uint<8>, mask: uint<1>
    // ---------------------------------------------------------------------------------
    // Read ports
    // CHECK-NEXT: %[[MEMORY_A_R_ADDR:.+]] = firrtl.subfield %[[MEMORY_A_R]](0)
    // CHECK-NEXT: firrtl.connect %[[MEMORY_A_R_ADDR]], %[[MEMORY_R_ADDR:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_R_ADDR:.+]] = firrtl.subfield %[[MEMORY_B_R]](0)
    // CHECK-NEXT: firrtl.connect %[[MEMORY_B_R_ADDR]], %[[MEMORY_R_ADDR]]
    // CHECK-NEXT: %[[MEMORY_A_R_EN:.+]] = firrtl.subfield %[[MEMORY_A_R]](1)
    // CHECK-NEXT: firrtl.connect %[[MEMORY_A_R_EN]], %[[MEMORY_R_EN:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_R_EN:.+]] = firrtl.subfield %[[MEMORY_B_R]](1)
    // CHECK-NEXT: firrtl.connect %[[MEMORY_B_R_EN]], %[[MEMORY_R_EN]]
    // CHECK-NEXT: %[[MEMORY_A_R_CLK:.+]] = firrtl.subfield %[[MEMORY_A_R]](2)
    // CHECK-NEXT: firrtl.connect %[[MEMORY_A_R_CLK]], %[[MEMORY_R_CLK:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_R_CLK:.+]] = firrtl.subfield %[[MEMORY_B_R]](2)
    // CHECK-NEXT: firrtl.connect %[[MEMORY_B_R_CLK]], %[[MEMORY_R_CLK]]
    // CHECK-NEXT: %[[MEMORY_A_R_DATA:.+]] = firrtl.subfield %[[MEMORY_A_R]](3)
    // CHECK-NEXT: firrtl.connect %[[WIRE_A_R_DATA:.+]], %[[MEMORY_A_R_DATA]] :
    // CHECK-NEXT: %[[MEMORY_B_R_DATA:.+]] = firrtl.subfield %[[MEMORY_B_R]](3)
    // CHECK-NEXT: firrtl.connect %[[WIRE_B_R_DATA:.+]], %[[MEMORY_B_R_DATA]] :
    // ---------------------------------------------------------------------------------
    // Write Ports
    // CHECK-NEXT: %[[MEMORY_A_W_ADDR:.+]] = firrtl.subfield %[[MEMORY_A_W]](0)
    // CHECK-NEXT: firrtl.connect %[[MEMORY_A_W_ADDR]], %[[MEMORY_W_ADDR:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_W_ADDR:.+]] = firrtl.subfield %[[MEMORY_B_W]](0)
    // CHECK-NEXT: firrtl.connect %[[MEMORY_B_W_ADDR]], %[[MEMORY_W_ADDR]] :
    // CHECK-NEXT: %[[MEMORY_A_W_EN:.+]] = firrtl.subfield %[[MEMORY_A_W]](1)
    // CHECK-NEXT: firrtl.connect %[[MEMORY_A_W_EN]], %[[MEMORY_W_EN:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_W_EN:.+]] = firrtl.subfield %[[MEMORY_B_W]](1)
    // CHECK-NEXT: firrtl.connect %[[MEMORY_B_W_EN]], %[[MEMORY_W_EN]] :
    // CHECK-NEXT: %[[MEMORY_A_W_CLK:.+]] = firrtl.subfield %[[MEMORY_A_W]](2)
    // CHECK-NEXT: firrtl.connect %[[MEMORY_A_W_CLK]], %[[MEMORY_W_CLK:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_W_CLK:.+]] = firrtl.subfield %[[MEMORY_B_W]](2)
    // CHECK-NEXT: firrtl.connect %[[MEMORY_B_W_CLK]], %[[MEMORY_W_CLK]] :
    // CHECK-NEXT: %[[MEMORY_A_W_DATA:.+]] = firrtl.subfield %[[MEMORY_A_W]](3)
    // CHECK-NEXT: firrtl.connect %[[MEMORY_A_W_DATA]], %[[WIRE_A_W_DATA:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_W_DATA:.+]] = firrtl.subfield %[[MEMORY_B_W]](3)
    // CHECK-NEXT: firrtl.connect %[[MEMORY_B_W_DATA]], %[[WIRE_B_W_DATA:.+]] :
    // CHECK-NEXT: %[[MEMORY_A_W_MASK:.+]] = firrtl.subfield %[[MEMORY_A_W]](4)
    // CHECK-NEXT: firrtl.connect %[[MEMORY_A_W_MASK]], %[[WIRE_A_W_MASK:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_W_MASK:.+]] = firrtl.subfield %[[MEMORY_B_W]](4)
    // CHECK-NEXT: firrtl.connect %[[MEMORY_B_W_MASK]], %[[WIRE_B_W_MASK:.+]] :
    //
    // Connections to module ports
    // CHECK-NEXT: firrtl.connect %[[MEMORY_R_CLK]], %clock
    // CHECK-NEXT: firrtl.connect %[[MEMORY_R_EN]], %rEn
    // CHECK-NEXT: firrtl.connect %[[MEMORY_R_ADDR]], %rAddr
    // CHECK-NEXT: firrtl.connect %rData_a, %[[WIRE_A_R_DATA]]
    // CHECK-NEXT: firrtl.connect %rData_b, %[[WIRE_B_R_DATA]]
    // CHECK-NEXT: firrtl.connect %[[MEMORY_W_CLK]], %clock
    // CHECK-NEXT: firrtl.connect %[[MEMORY_W_EN]], %wEn
    // CHECK-NEXT: firrtl.connect %[[MEMORY_W_ADDR]], %wAddr
    // CHECK-NEXT: firrtl.connect %[[WIRE_A_W_MASK]], %wMask_a
    // CHECK-NEXT: firrtl.connect %[[WIRE_B_W_MASK]], %wMask_b
    // CHECK-NEXT: firrtl.connect %[[WIRE_A_W_DATA]], %wData_a
    // CHECK-NEXT: firrtl.connect %[[WIRE_B_W_DATA]], %wData_b

    // ---------------------------------------------------------------------------------
    // If flatten memory data is enabled
    // FLATTEN: %[[memory_r:.+]], %[[memory_w:.+]] = firrtl.mem Undefined  {depth = 16 : i64, name = "memory", portNames = ["r", "w"], readLatency = 0 : i32, writeLatency = 1 : i32} 
    // FLATTEN-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<16>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<16>, mask: uint<2>>
    // FLATTEN: %[[v0:.+]] = firrtl.subfield %[[memory_r]](0)
    // FLATTEN: firrtl.connect %[[v0]], %[[memory_r_addr:.+]] :
    // FLATTEN: %[[v1:.+]] = firrtl.subfield %[[memory_r]](1)
    // FLATTEN: firrtl.connect %[[v1]], %[[memory_r_en:.+]] :
    // FLATTEN: %[[v2:.+]] = firrtl.subfield %[[memory_r]](2)
    // FLATTEN: firrtl.connect %[[v2]], %[[memory_r_clk:.+]] :
    // FLATTEN: %[[v3:.+]] = firrtl.subfield %[[memory_r]](3)
    //
    // ---------------------------------------------------------------------------------
    // Read ports
    // FLATTEN:  %[[v4:.+]] = firrtl.bits %[[v3]] 7 to 0 :
    // FLATTEN:  %[[v5:.+]] = firrtl.bits %[[v3]] 15 to 8 :
    // FLATTEN:  firrtl.connect %[[memory_r_data_a:.+]], %[[v4]] :
    // FLATTEN:  firrtl.connect %[[memory_r_data_b:.+]], %[[v5]] :
    // --------------------------------------------------------------------------------
    // Write Ports
    // FLATTEN:  %[[v9:.+]] = firrtl.subfield %[[memory_w]](3)
    // FLATTEN:  %[[v10:.+]] = firrtl.cat %[[memory_w_data_b:.+]], %[[memory_w_data_a:.+]] :
    // FLATTEN:  firrtl.connect %[[v9]], %[[v10]]
    //
    // --------------------------------------------------------------------------------
    // Mask Ports
    //  FLATTEN: %[[v11:.+]] = firrtl.subfield %[[memory_w]](4)
    //  FLATTEN: %[[v12:.+]] = firrtl.cat %[[memory_w_mask_b:.+]], %[[memory_w_mask_a:.+]] :
    //  FLATTEN: firrtl.connect %[[v11]], %[[v12]]
    // --------------------------------------------------------------------------------
    // Connections to module ports
    // FLATTEN:  firrtl.connect %[[memory_r_clk]], %clock
    // FLATTEN:  firrtl.connect %[[memory_r_en]], %rEn
    // FLATTEN:  firrtl.connect %[[memory_r_addr]], %rAddr
    // FLATTEN:  firrtl.connect %rData_a, %[[memory_r_data_a]] : !firrtl.uint<8>, !firrtl.uint<8>
    // FLATTEN:  firrtl.connect %rData_b, %[[memory_r_data_b]] : !firrtl.uint<8>, !firrtl.uint<8>
    // FLATTEN:  firrtl.connect %[[memory_w_mask_a]], %wMask_a : !firrtl.uint<1>, !firrtl.uint<1>
    // FLATTEN:  firrtl.connect %[[memory_w_mask_b]], %wMask_b : !firrtl.uint<1>, !firrtl.uint<1>
    // FLATTEN:  firrtl.connect %[[memory_w_data_a]], %wData_a : !firrtl.uint<8>, !firrtl.uint<8>
    // FLATTEN:  firrtl.connect %[[memory_w_data_b]], %wData_b : !firrtl.uint<8>, !firrtl.uint<8>
  }

  firrtl.module @Mem2_flatten(in %clock: !firrtl.clock, in %rAddr: !firrtl.uint<4>, in %rEn: !firrtl.uint<1>, out
  %rData: !firrtl.bundle<a: uint<8>, b: uint<16>>, in %wAddr: !firrtl.uint<4>, in %wEn: !firrtl.uint<1>, in %wMask:
  !firrtl.bundle<a: uint<1>, b: uint<1>>, in %wData: !firrtl.bundle<a: uint<8>, b: uint<16>>) {
    %memory_r, %memory_w = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r", "w"], readLatency
    = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a:
    uint<8>, b: uint<16>>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<16>>, mask: bundle<a: uint<1>, b: uint<1>>>
    %0 = firrtl.subfield %memory_r(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a:
    uint<8>, b: uint<16>>>) -> !firrtl.clock
    firrtl.connect %0, %clock : !firrtl.clock, !firrtl.clock
    %1 = firrtl.subfield %memory_r(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<16>>>) -> !firrtl.uint<1>
    firrtl.connect %1, %rEn : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.subfield %memory_r(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<16>>>) -> !firrtl.uint<4>
    firrtl.connect %2, %rAddr : !firrtl.uint<4>, !firrtl.uint<4>
    %3 = firrtl.subfield %memory_r(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a:
    uint<8>, b: uint<16>>>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    firrtl.connect %rData, %3 : !firrtl.bundle<a: uint<8>, b: uint<16>>, !firrtl.bundle<a: uint<8>, b: uint<16>>
    %4 = firrtl.subfield %memory_w(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<16>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.clock
    firrtl.connect %4, %clock : !firrtl.clock, !firrtl.clock
    %5 = firrtl.subfield %memory_w(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<16>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<1>
    firrtl.connect %5, %wEn : !firrtl.uint<1>, !firrtl.uint<1>
    %6 = firrtl.subfield %memory_w(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<16>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.uint<4>
    firrtl.connect %6, %wAddr : !firrtl.uint<4>, !firrtl.uint<4>
    %7 = firrtl.subfield %memory_w(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<16>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.connect %7, %wMask : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
    %8 = firrtl.subfield %memory_w(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<16>>, mask: bundle<a: uint<1>, b: uint<1>>>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    firrtl.connect %8, %wData : !firrtl.bundle<a: uint<8>, b: uint<16>>, !firrtl.bundle<a: uint<8>, b: uint<16>>

    // FLATTEN:  %[[memory_r:.+]], %[[memory_w:.+]] = firrtl.mem Undefined  {depth = 16 : i64, name = "memory", portNames = ["r", "w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<24>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<24>, mask: uint<3>>
    // FLATTEN:  %[[v11:.+]] = firrtl.subfield %[[memory_w]](4)
    // FLATTENsss  %[[v12:.+]] = firrtl.cat %[[memory_w_mask_b]], %[[memory_w_mask_a]] :
    // FLATTEN:  %[[v13:.+]] = firrtl.bits %[[v12]] 0 to 0
    // FLATTEN:  %[[v14:.+]] = firrtl.bits %[[v12]] 1 to 1
    // FLATTEN:  %[[v15:.+]] = firrtl.cat %[[v14]], %[[v13]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    // FLATTEN:  %[[v16:.+]] = firrtl.cat %[[v14]], %[[v15]] : (!firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<3>
    // FLATTEN:  firrtl.connect %[[v11]], %[[v16]] : !firrtl.uint<3>, !firrtl.uint<3>
    // FLATTEN:  firrtl.connect %[[memory_w_mask_b]], %wMask_b : !firrtl.uint<1>, !firrtl.uint<1>
    // FLATTEN:  firrtl.connect %[[memory_w_data_a]], %wData_a : !firrtl.uint<8>, !firrtl.uint<8>

  }
// Test that a memory with a readwrite port is split into 1r1w
//
// circuit Foo:
//  module Foo:
//    input clock: Clock
//    input rwEn: UInt<1>
//    input rwMode: UInt<1>
//    input rwAddr: UInt<4>
//    input rwMask: UInt<1>
//    input rwDataIn: UInt<8>
//    output rwDataOut: UInt<8>
//
//    mem memory:
//      data-type => UInt<8>
//      depth => 16
//      readwriter => rw
//      read-latency => 0
//      write-latency => 1
//      read-under-write => undefined
//
//    memory.rw.clk <= clock
//    memory.rw.en <= rwEn
//    memory.rw.addr <= rwAddr
//    memory.rw.wmode <= rwMode
//    memory.rw.wmask <= rwMask
//    memory.rw.wdata <= rwDataIn
//    rwDataOut <= memory.rw.rdata

  firrtl.module @MemoryRWSplit(in %clock: !firrtl.clock, in %rwEn: !firrtl.uint<1>, in %rwMode: !firrtl.uint<1>, in %rwAddr: !firrtl.uint<4>, in %rwMask: !firrtl.uint<1>, in %rwDataIn: !firrtl.uint<8>, out %rwDataOut: !firrtl.uint<8>) {
    %memory_rw = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["rw"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<1>>
    %0 = firrtl.subfield %memory_rw(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<1>>) -> !firrtl.clock
    firrtl.connect %0, %clock : !firrtl.clock, !firrtl.clock
    %1 = firrtl.subfield %memory_rw(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %1, %rwEn : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.subfield %memory_rw(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<1>>) -> !firrtl.uint<4>
    firrtl.connect %2, %rwAddr : !firrtl.uint<4>, !firrtl.uint<4>
    %3 = firrtl.subfield %memory_rw(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %3, %rwMode : !firrtl.uint<1>, !firrtl.uint<1>
    %4 = firrtl.subfield %memory_rw(6) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %4, %rwMask : !firrtl.uint<1>, !firrtl.uint<1>
    %5 = firrtl.subfield %memory_rw(5) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<1>>) -> !firrtl.uint<8>
    firrtl.connect %5, %rwDataIn : !firrtl.uint<8>, !firrtl.uint<8>
    %6 = firrtl.subfield %memory_rw(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<1>>) -> !firrtl.uint<8>
    firrtl.connect %rwDataOut, %6 : !firrtl.uint<8>, !firrtl.uint<8>
  }




  firrtl.module @MemoryRWSplitUnique() {
    %memory_rw, %memory_rw_r, %memory_rw_w = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["rw", "rw_r", "rw_w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<1>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
  }


// https://github.com/llvm/circt/issues/593

    firrtl.module @mod_2(in %clock: !firrtl.clock, in %inp_a: !firrtl.bundle<inp_d: uint<14>>) {
    }
    firrtl.module @top_mod(in %clock: !firrtl.clock) {
      %U0_clock, %U0_inp_a = firrtl.instance U0 @mod_2(in clock: !firrtl.clock, in inp_a: !firrtl.bundle<inp_d: uint<14>>)
      %0 = firrtl.invalidvalue : !firrtl.clock
      firrtl.connect %U0_clock, %0 : !firrtl.clock, !firrtl.clock
      %1 = firrtl.invalidvalue : !firrtl.bundle<inp_d: uint<14>>
      firrtl.connect %U0_inp_a, %1 : !firrtl.bundle<inp_d: uint<14>>, !firrtl.bundle<inp_d: uint<14>>
    }



//CHECK-LABEL:     firrtl.module @mod_2(in %clock: !firrtl.clock, in %inp_a_inp_d: !firrtl.uint<14>)
//CHECK:    firrtl.module @top_mod(in %clock: !firrtl.clock)
//CHECK-NEXT:      %U0_clock, %U0_inp_a_inp_d = firrtl.instance U0 @mod_2(in clock: !firrtl.clock, in inp_a_inp_d: !firrtl.uint<14>)
//CHECK-NEXT:      %invalid_clock = firrtl.invalidvalue : !firrtl.clock
//CHECK-NEXT:      firrtl.connect %U0_clock, %invalid_clock : !firrtl.clock, !firrtl.clock
//CHECK-NEXT:      %invalid_ui14 = firrtl.invalidvalue : !firrtl.uint<14>
//CHECK-NEXT:      firrtl.connect %U0_inp_a_inp_d, %invalid_ui14 : !firrtl.uint<14>, !firrtl.uint<14>

//AGGREGATE-LABEL: firrtl.module @mod_2(in %clock: !firrtl.clock, in %inp_a: !firrtl.bundle<inp_d: uint<14>>)
//AGGREGATE:    firrtl.module @top_mod(in %clock: !firrtl.clock)
//AGGREGATE-NEXT:  %U0_clock, %U0_inp_a = firrtl.instance U0  @mod_2(in clock: !firrtl.clock, in inp_a: !firrtl.bundle<inp_d: uint<14>>)
//AGGREGATE-NEXT:  %invalid_clock = firrtl.invalidvalue : !firrtl.clock
//AGGREGATE-NEXT:  firrtl.connect %U0_clock, %invalid_clock : !firrtl.clock, !firrtl.clock
//AGGREGATE-NEXT:  %invalid = firrtl.invalidvalue : !firrtl.bundle<inp_d: uint<14>>
//AGGREGATE-NEXT:  %0 = firrtl.subfield %invalid(0) : (!firrtl.bundle<inp_d: uint<14>>) -> !firrtl.uint<14>
//AGGREGATE-NEXT:  %1 = firrtl.subfield %U0_inp_a(0) : (!firrtl.bundle<inp_d: uint<14>>) -> !firrtl.uint<14>
//AGGREGATE-NEXT:  firrtl.connect %1, %0 : !firrtl.uint<14>, !firrtl.uint<14>
// https://github.com/llvm/circt/issues/661

// This test is just checking that the following doesn't error.
    // COMMON-LABEL: firrtl.module @Issue661
    firrtl.module @Issue661(in %clock: !firrtl.clock) {
      %head_MPORT_2, %head_MPORT_6 = firrtl.mem Undefined {depth = 20 : i64, name = "head", portNames = ["MPORT_2", "MPORT_6"], readLatency = 0 : i32, writeLatency = 1 : i32}
      : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<5>, mask: uint<1>>,
        !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<5>, mask: uint<1>>
      %127 = firrtl.subfield %head_MPORT_6(2) : (!firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<5>, mask: uint<1>>) -> !firrtl.clock
    }

// Check that a non-bundled mux ops are untouched.
    // CHECK-LABEL: firrtl.module @Mux
    firrtl.module @Mux(in %p: !firrtl.uint<1>, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
      // CHECK-NEXT: %0 = firrtl.mux(%p, %a, %b) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %c, %0 : !firrtl.uint<1>, !firrtl.uint<1>
      %0 = firrtl.mux(%p, %a, %b) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      firrtl.connect %c, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    }
    // CHECK-LABEL: firrtl.module @MuxBundle
    firrtl.module @MuxBundle(in %p: !firrtl.uint<1>, in %a: !firrtl.bundle<a: uint<1>>, in %b: !firrtl.bundle<a: uint<1>>, out %c: !firrtl.bundle<a: uint<1>>) {
      // CHECK-NEXT: %0 = firrtl.mux(%p, %a_a, %b_a) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %c_a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
      %0 = firrtl.mux(%p, %a, %b) : (!firrtl.uint<1>, !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>) -> !firrtl.bundle<a: uint<1>>
      firrtl.connect %c, %0 : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    }

    // CHECK-LABEL: firrtl.module @NodeBundle
    firrtl.module @NodeBundle(in %a: !firrtl.bundle<a: uint<1>>, out %b: !firrtl.uint<1>) {
      // CHECK-NEXT: %n_a = firrtl.node %a_a  : !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %b, %n_a : !firrtl.uint<1>, !firrtl.uint<1>
      %n = firrtl.node %a : !firrtl.bundle<a: uint<1>>
      %n_a = firrtl.subfield %n(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %b, %n_a : !firrtl.uint<1>, !firrtl.uint<1>
    }

    // CHECK-LABEL: firrtl.module @RegBundle(in %a_a: !firrtl.uint<1>, in %clk: !firrtl.clock, out %b_a: !firrtl.uint<1>)
    firrtl.module @RegBundle(in %a: !firrtl.bundle<a: uint<1>>, in %clk: !firrtl.clock, out %b: !firrtl.bundle<a: uint<1>>) {
      // CHECK-NEXT: %x_a = firrtl.reg %clk : !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %x_a, %a_a : !firrtl.uint<1>, !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %b_a, %x_a : !firrtl.uint<1>, !firrtl.uint<1>
      %x = firrtl.reg %clk {name = "x"} : !firrtl.bundle<a: uint<1>>
      %0 = firrtl.subfield %x(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
      %1 = firrtl.subfield %a(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %0, %1 : !firrtl.uint<1>, !firrtl.uint<1>
      %2 = firrtl.subfield %b(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
      %3 = firrtl.subfield %x(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %2, %3 : !firrtl.uint<1>, !firrtl.uint<1>
    }

    // CHECK-LABEL: firrtl.module @RegBundleWithBulkConnect(in %a_a: !firrtl.uint<1>, in %clk: !firrtl.clock, out %b_a: !firrtl.uint<1>)
    firrtl.module @RegBundleWithBulkConnect(in %a: !firrtl.bundle<a: uint<1>>, in %clk: !firrtl.clock, out %b: !firrtl.bundle<a: uint<1>>) {
      // CHECK-NEXT: %x_a = firrtl.reg %clk : !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %x_a, %a_a : !firrtl.uint<1>, !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %b_a, %x_a : !firrtl.uint<1>, !firrtl.uint<1>
      %x = firrtl.reg %clk {name = "x"} : !firrtl.bundle<a: uint<1>>
      firrtl.connect %x, %a : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
      firrtl.connect %b, %x : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    }

    // CHECK-LABEL: firrtl.module @WireBundle(in %a_a: !firrtl.uint<1>,  out %b_a: !firrtl.uint<1>)
    firrtl.module @WireBundle(in %a: !firrtl.bundle<a: uint<1>>,  out %b: !firrtl.bundle<a: uint<1>>) {
      // CHECK-NEXT: %x_a = firrtl.wire  : !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %x_a, %a_a : !firrtl.uint<1>, !firrtl.uint<1>
      // CHECK-NEXT: firrtl.connect %b_a, %x_a : !firrtl.uint<1>, !firrtl.uint<1>
      %x = firrtl.wire : !firrtl.bundle<a: uint<1>>
      %0 = firrtl.subfield %x(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
      %1 = firrtl.subfield %a(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %0, %1 : !firrtl.uint<1>, !firrtl.uint<1>
      %2 = firrtl.subfield %b(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
      %3 = firrtl.subfield %x(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %2, %3 : !firrtl.uint<1>, !firrtl.uint<1>
    }

  // CHECK-LABEL: firrtl.module @WireBundlesWithBulkConnect
  firrtl.module @WireBundlesWithBulkConnect(in %source: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
                             out %sink: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
    // CHECK: %w_valid = firrtl.wire  : !firrtl.uint<1>
    // CHECK: %w_ready = firrtl.wire  : !firrtl.uint<1>
    // CHECK: %w_data = firrtl.wire  : !firrtl.uint<64>
    %w = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    // CHECK: firrtl.connect %w_valid, %source_valid : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.connect %source_ready, %w_ready : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.connect %w_data, %source_data : !firrtl.uint<64>, !firrtl.uint<64>
    firrtl.connect %w, %source : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    // CHECK: firrtl.connect %sink_valid, %w_valid : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.connect %w_ready, %sink_ready : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.connect %sink_data, %w_data : !firrtl.uint<64>, !firrtl.uint<64>
    firrtl.connect %sink, %w : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
  }

// Test vector lowering
  firrtl.module @LowerVectors(in %a: !firrtl.vector<uint<1>, 2>, out %b: !firrtl.vector<uint<1>, 2>) {
    firrtl.connect %b, %a: !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
  // CHECK-LABEL: firrtl.module @LowerVectors(in %a_0: !firrtl.uint<1>, in %a_1: !firrtl.uint<1>, out %b_0: !firrtl.uint<1>, out %b_1: !firrtl.uint<1>)
  // CHECK: firrtl.connect %b_0, %a_0
  // CHECK: firrtl.connect %b_1, %a_1
  // AGGREGATE-LABEL: firrtl.module @LowerVectors(in %a: !firrtl.vector<uint<1>, 2>, out %b: !firrtl.vector<uint<1>, 2>)
  // AGGREGATE-NEXT: %0 = firrtl.subindex %a[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT: %1 = firrtl.subindex %b[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT: firrtl.connect %1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  // AGGREGATE-NEXT: %2 = firrtl.subindex %a[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT: %3 = firrtl.subindex %b[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT: firrtl.connect %3, %2 : !firrtl.uint<1>, !firrtl.uint<1>

// Test vector of bundles lowering
  // COMMON-LABEL: firrtl.module @LowerVectorsOfBundles(in %in_0_a: !firrtl.uint<1>, out %in_0_b: !firrtl.uint<1>, in %in_1_a: !firrtl.uint<1>, out %in_1_b: !firrtl.uint<1>, out %out_0_a: !firrtl.uint<1>, in %out_0_b: !firrtl.uint<1>, out %out_1_a: !firrtl.uint<1>, in %out_1_b: !firrtl.uint<1>)
  firrtl.module @LowerVectorsOfBundles(in %in: !firrtl.vector<bundle<a : uint<1>, b  flip: uint<1>>, 2>,
                                       out %out: !firrtl.vector<bundle<a : uint<1>, b  flip: uint<1>>, 2>) {
    // COMMON:      firrtl.connect %out_0_a, %in_0_a : !firrtl.uint<1>, !firrtl.uint<1>
    // COMMON-NEXT: firrtl.connect %in_0_b, %out_0_b : !firrtl.uint<1>, !firrtl.uint<1>
    // COMMON-NEXT: firrtl.connect %out_1_a, %in_1_a : !firrtl.uint<1>, !firrtl.uint<1>
    // COMMON-NEXT: firrtl.connect %in_1_b, %out_1_b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %in: !firrtl.vector<bundle<a : uint<1>, b flip: uint<1>>, 2>, !firrtl.vector<bundle<a : uint<1>, b flip: uint<1>>, 2>
  }

  // COMMON-LABEL: firrtl.extmodule @ExternalModule(in source_valid: !firrtl.uint<1>, out source_ready: !firrtl.uint<1>, in source_data: !firrtl.uint<64>)
  firrtl.extmodule @ExternalModule(in source: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
  firrtl.module @Test() {
    // COMMON: %inst_source_valid, %inst_source_ready, %inst_source_data = firrtl.instance "" @ExternalModule(in source_valid: !firrtl.uint<1>, out source_ready: !firrtl.uint<1>, in source_data: !firrtl.uint<64>)
    %inst_source = firrtl.instance "" @ExternalModule(in source: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
  }

// Test RegResetOp lowering
  // CHECK-LABEL: firrtl.module @LowerRegResetOp
  firrtl.module @LowerRegResetOp(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a_d: !firrtl.vector<uint<1>, 2>, out %a_q: !firrtl.vector<uint<1>, 2>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %init = firrtl.wire  : !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subindex %init[0] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = firrtl.subindex %init[1] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %r = firrtl.regreset %clock, %reset, %init {name = "r"} : !firrtl.uint<1>, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    firrtl.connect %r, %a_d : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    firrtl.connect %a_q, %r : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %init_0 = firrtl.wire  : !firrtl.uint<1>
  // CHECK:   %init_1 = firrtl.wire  : !firrtl.uint<1>
  // CHECK:   firrtl.connect %init_0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.connect %init_1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   %r_0 = firrtl.regreset %clock, %reset, %init_0 : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   %r_1 = firrtl.regreset %clock, %reset, %init_1 : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.connect %r_0, %a_d_0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.connect %r_1, %a_d_1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.connect %a_q_0, %r_0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.connect %a_q_1, %r_1 : !firrtl.uint<1>, !firrtl.uint<1>
  // AGGREGATE:       %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // AGGREGATE-NEXT:  %init = firrtl.wire  : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  %0 = firrtl.subindex %init[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // AGGREGATE-NEXT:  %1 = firrtl.subindex %init[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  firrtl.connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // AGGREGATE-NEXT:  %r = firrtl.regreset %clock, %reset, %init  : !firrtl.uint<1>, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  %2 = firrtl.subindex %a_d[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  %3 = firrtl.subindex %r[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  firrtl.connect %3, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  // AGGREGATE-NEXT:  %4 = firrtl.subindex %a_d[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  %5 = firrtl.subindex %r[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  firrtl.connect %5, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  // AGGREGATE-NEXT:  %6 = firrtl.subindex %r[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  %7 = firrtl.subindex %a_q[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  firrtl.connect %7, %6 : !firrtl.uint<1>, !firrtl.uint<1>
  // AGGREGATE-NEXT:  %8 = firrtl.subindex %r[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  %9 = firrtl.subindex %a_q[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  firrtl.connect %9, %8 : !firrtl.uint<1>, !firrtl.uint<1>

// Test RegResetOp lowering without name attribute
// https://github.com/llvm/circt/issues/795
  // CHECK-LABEL: firrtl.module @LowerRegResetOpNoName
  firrtl.module @LowerRegResetOpNoName(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a_d: !firrtl.vector<uint<1>, 2>, out %a_q: !firrtl.vector<uint<1>, 2>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %init = firrtl.wire  : !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subindex %init[0] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = firrtl.subindex %init[1] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %r = firrtl.regreset %clock, %reset, %init {name = ""} : !firrtl.uint<1>, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    firrtl.connect %r, %a_d : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    firrtl.connect %a_q, %r : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %init_0 = firrtl.wire  : !firrtl.uint<1>
  // CHECK:   %init_1 = firrtl.wire  : !firrtl.uint<1>
  // CHECK:   firrtl.connect %init_0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.connect %init_1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   %0 = firrtl.regreset %clock, %reset, %init_0 : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   %1 = firrtl.regreset %clock, %reset, %init_1 : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.connect %0, %a_d_0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.connect %1, %a_d_1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.connect %a_q_0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   firrtl.connect %a_q_1, %1 : !firrtl.uint<1>, !firrtl.uint<1>

// Test RegOp lowering without name attribute
// https://github.com/llvm/circt/issues/795
  // CHECK-LABEL: firrtl.module @lowerRegOpNoName
  firrtl.module @lowerRegOpNoName(in %clock: !firrtl.clock, in %a_d: !firrtl.vector<uint<1>, 2>, out %a_q: !firrtl.vector<uint<1>, 2>) {
    %r = firrtl.reg %clock {name = ""} : !firrtl.vector<uint<1>, 2>
      firrtl.connect %r, %a_d : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
      firrtl.connect %a_q, %r : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
 // CHECK:    %0 = firrtl.reg %clock : !firrtl.uint<1>
 // CHECK:    %1 = firrtl.reg %clock : !firrtl.uint<1>
 // CHECK:    firrtl.connect %0, %a_d_0 : !firrtl.uint<1>, !firrtl.uint<1>
 // CHECK:    firrtl.connect %1, %a_d_1 : !firrtl.uint<1>, !firrtl.uint<1>
 // CHECK:    firrtl.connect %a_q_0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
 // CHECK:    firrtl.connect %a_q_1, %1 : !firrtl.uint<1>, !firrtl.uint<1>

// Test that InstanceOp Annotations are copied to the new instance.
  firrtl.module @Bar(out %a: !firrtl.vector<uint<1>, 2>) {
    %0 = firrtl.invalidvalue : !firrtl.vector<uint<1>, 2>
    firrtl.connect %a, %0 : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
  firrtl.module @AnnotationsInstanceOp() {
    %bar_a = firrtl.instance bar {annotations = [{a = "a"}]} @Bar(out a: !firrtl.vector<uint<1>, 2>)
  }
  // CHECK: firrtl.instance
  // CHECK-SAME: annotations = [{a = "a"}]

// Test that MemOp Annotations are copied to lowered MemOps.
  // COMMON-LABEL: firrtl.module @AnnotationsMemOp
  firrtl.module @AnnotationsMemOp() {
    %bar_r, %bar_w = firrtl.mem Undefined  {annotations = [{a = "a"}], depth = 16 : i64, name = "bar", portNames = ["r", "w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: vector<uint<8>, 2>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: vector<uint<8>, 2>, mask: vector<uint<1>, 2>>
  }
  // COMMON: firrtl.mem
  // COMMON-SAME: annotations = [{a = "a"}]
  // COMMON: firrtl.mem
  // COMMON-SAME: annotations = [{a = "a"}]

// Test that WireOp Annotations are copied to lowered WireOps.
  // CHECK-LABEL: firrtl.module @AnnotationsWireOp
  firrtl.module @AnnotationsWireOp() {
    %bar = firrtl.wire  {annotations = [{a = "a"}]} : !firrtl.vector<uint<1>, 2>
  }
  // CHECK: firrtl.wire
  // CHECK-SAME: annotations = [{a = "a"}]
  // CHECK: firrtl.wire
  // CHECK-SAME: annotations = [{a = "a"}]

// Test that WireOp annotations which are sensitive to field IDs are annotated
// with the lowered field IDs.
  // COMMON-LABEL: firrtl.module @AnnotationsWithFieldIdWireOp
  firrtl.module @AnnotationsWithFieldIdWireOp() {
    %foo = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation"}]} : !firrtl.uint<1>
    %bar = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation"}]} : !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<1>>
    %baz = firrtl.wire {annotations = [#firrtl<"subAnno<fieldID = 2, {class = \"sifive.enterprise.grandcentral.SignalDriverAnnotation\"}>">]} : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
  }
  // CHECK: %foo = firrtl.wire
  // CHECK-SAME: {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation"}
  // CHECK: %bar_a_0 = firrtl.wire
  // CHECK-SAME: {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", fieldID = 2 : i64}
  // CHECK: %bar_a_1 = firrtl.wire
  // CHECK-SAME: {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", fieldID = 3 : i64}
  // CHECK: %bar_b = firrtl.wire
  // CHECK-SAME: {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", fieldID = 4 : i64}
  // CHECK: %baz_a = firrtl.wire
  // CHECK-NOT:  {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation"}
  // CHECK: %baz_b_0 = firrtl.wire
  // CHECK-SAME: {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", fieldID = 1 : i64}
  // CHECK: %baz_b_1 = firrtl.wire
  // CHECK-SAME: {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", fieldID = 2 : i64}
  // AGGREGATE:  %baz = firrtl.wire
  // AGGREGATE-SAME: {annotations = [#firrtl.subAnno<fieldID = 2, {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation"}>]}

// Test that Reg/RegResetOp Annotations are copied to lowered registers.
  // CHECK-LABEL: firrtl.module @AnnotationsRegOp
  firrtl.module @AnnotationsRegOp(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %bazInit = firrtl.wire  : !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subindex %bazInit[0] : !firrtl.vector<uint<1>, 2>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = firrtl.subindex %bazInit[1] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %bar = firrtl.reg %clock  {annotations = [{a = "a"}], name = "bar"} : !firrtl.vector<uint<1>, 2>
    %baz = firrtl.regreset %clock, %reset, %bazInit  {annotations = [{b = "b"}], name = "baz"} : !firrtl.uint<1>, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
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
// CHECK-LABEL: firrtl.module @WhenOp
  firrtl.module @WhenOp (in %p: !firrtl.uint<1>,
                         in %in : !firrtl.bundle<a: uint<1>, b: uint<1>>,
                         out %out : !firrtl.bundle<a: uint<1>, b: uint<1>>) {
    // No else region.
    firrtl.when %p {
      // CHECK: firrtl.connect %out_a, %in_a : !firrtl.uint<1>, !firrtl.uint<1>
      // CHECK: firrtl.connect %out_b, %in_b : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %out, %in : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
    }

    // Else region.
    firrtl.when %p {
      // CHECK: firrtl.connect %out_a, %in_a : !firrtl.uint<1>, !firrtl.uint<1>
      // CHECK: firrtl.connect %out_b, %in_b : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %out, %in : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
    } else {
      // CHECK: firrtl.connect %out_a, %in_a : !firrtl.uint<1>, !firrtl.uint<1>
      // CHECK: firrtl.connect %out_b, %in_b : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %out, %in : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
    }
  }

// Test that subfield annotations on wire are lowred to appropriate instance based on fieldID.
  // CHECK-LABEL: firrtl.module @AnnotationsBundle
  firrtl.module @AnnotationsBundle() {
    %bar = firrtl.wire  {annotations = [#firrtl.subAnno<fieldID = 3, {one}>, #firrtl.subAnno<fieldID = 5, {two}>]} : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>

      // TODO: Enable this
      // CHECK: %bar_0_baz = firrtl.wire  : !firrtl.uint<1>
      // CHECK: %bar_0_qux = firrtl.wire {annotations = [{one}]} : !firrtl.uint<1>
      // CHECK: %bar_1_baz = firrtl.wire {annotations = [{two}]} : !firrtl.uint<1>
      // CHECK: %bar_1_qux = firrtl.wire  : !firrtl.uint<1>

    %quux = firrtl.wire  {annotations = [#firrtl.subAnno<fieldID = 0, {zero}>]} : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
      // CHECK: %quux_0_baz = firrtl.wire {annotations = [{zero}]} : !firrtl.uint<1>
      // CHECK: %quux_0_qux = firrtl.wire {annotations = [{zero}]} : !firrtl.uint<1>
      // CHECK: %quux_1_baz = firrtl.wire {annotations = [{zero}]} : !firrtl.uint<1>
      // CHECK: %quux_1_qux = firrtl.wire {annotations = [{zero}]} : !firrtl.uint<1>
  }

// Test that subfield annotations on reg are lowred to appropriate instance based on fieldID.
 // CHECK-LABEL: firrtl.module @AnnotationsBundle2
  firrtl.module @AnnotationsBundle2(in %clock: !firrtl.clock) {
    %bar = firrtl.reg %clock  {annotations = [#firrtl.subAnno<fieldID = 3, {one}>, #firrtl.subAnno<fieldID = 5, {two}>]} : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>

    // TODO: Enable this
    // CHECK: %bar_0_baz = firrtl.reg %clock  : !firrtl.uint<1>
    // CHECK: %bar_0_qux = firrtl.reg %clock  {annotations = [{one}]} : !firrtl.uint<1>
    // CHECK: %bar_1_baz = firrtl.reg %clock  {annotations = [{two}]} : !firrtl.uint<1>
    // CHECK: %bar_1_qux = firrtl.reg %clock  : !firrtl.uint<1>
  }

// Test that subfield annotations on reg are lowred to appropriate instance based on fieldID. Ignore un-flattened array targets
// circuit Foo: %[[{"one":null,"target":"~Foo|Foo>bar[0].qux[0]"},{"two":null,"target":"~Foo|Foo>bar[1].baz"},{"three":null,"target":"~Foo|Foo>bar[0].yes"} ]]

 // CHECK-LABEL: firrtl.module @AnnotationsBundle3
  firrtl.module @AnnotationsBundle3(in %clock: !firrtl.clock) {
    %bar = firrtl.reg %clock  {annotations = [#firrtl.subAnno<fieldID = 6, {one}>, #firrtl.subAnno<fieldID = 12, {two}>, #firrtl.subAnno<fieldID = 8, {three}>]} : !firrtl.vector<bundle<baz: vector<uint<1>, 2>, qux: vector<uint<1>, 2>, yes: bundle<a: uint<1>, b: uint<1>>>, 2>

    // TODO: Enable this
    // CHECK: %bar_0_baz_0 = firrtl.reg %clock  : !firrtl.uint<1>
    // CHECK: %bar_0_baz_1 = firrtl.reg %clock  : !firrtl.uint<1>
    // CHECK: %bar_0_qux_0 = firrtl.reg %clock  {annotations = [{one}]} : !firrtl.uint<1>
    // CHECK: %bar_0_qux_1 = firrtl.reg %clock  : !firrtl.uint<1>
    // CHECK: %bar_0_yes_a = firrtl.reg %clock  {annotations = [{three}]} : !firrtl.uint<1>
    // CHECK: %bar_0_yes_b = firrtl.reg %clock  {annotations = [{three}]} : !firrtl.uint<1>
    // CHECK: %bar_1_baz_0 = firrtl.reg %clock  {annotations = [{two}]} : !firrtl.uint<1>
    // CHECK: %bar_1_baz_1 = firrtl.reg %clock  {annotations = [{two}]} : !firrtl.uint<1>
    // CHECK: %bar_1_qux_0 = firrtl.reg %clock  : !firrtl.uint<1>
    // CHECK: %bar_1_qux_1 = firrtl.reg %clock  : !firrtl.uint<1>
    // CHECK: %bar_1_yes_a = firrtl.reg %clock  : !firrtl.uint<1>
    // CHECK: %bar_1_yes_b = firrtl.reg %clock  : !firrtl.uint<1>
  }

// Test wire connection semantics.  Based on the flippedness of the destination
// type, the connection may be reversed.
// CHECK-LABEL firrtl.module @WireSemantics
  firrtl.module @WireSemantics() {
    %a = firrtl.wire  : !firrtl.bundle<a: bundle<a: uint<1>>>
    %ax = firrtl.wire  : !firrtl.bundle<a: bundle<a: uint<1>>>
    // CHECK:  %a_a_a = firrtl.wire
    // CHECK-NEXT:  %ax_a_a = firrtl.wire
    firrtl.connect %a, %ax : !firrtl.bundle<a: bundle<a: uint<1>>>, !firrtl.bundle<a: bundle<a: uint<1>>>
    firrtl.partialconnect %a, %ax : !firrtl.bundle<a: bundle<a: uint<1>>>, !firrtl.bundle<a: bundle<a: uint<1>>>
    // a <= ax
    // CHECK-NEXT: firrtl.connect %a_a_a, %ax_a_a
    // a <- ax
    // CHECK-NEXT: firrtl.connect %a_a_a, %ax_a_a
    %0 = firrtl.subfield %a(0) : (!firrtl.bundle<a: bundle<a: uint<1>>>) -> !firrtl.bundle<a: uint<1>>
    %1 = firrtl.subfield %ax(0) : (!firrtl.bundle<a: bundle<a: uint<1>>>) -> !firrtl.bundle<a: uint<1>>
    firrtl.connect %0, %1 : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    firrtl.partialconnect %0, %1 : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    // a.a <= ax.a
    // CHECK: firrtl.connect %a_a_a, %ax_a_a
    // a.a <- ax.a
    // CHECK-NEXT: firrtl.connect %a_a_a, %ax_a_a
    %2 = firrtl.subfield %a(0) : (!firrtl.bundle<a: bundle<a: uint<1>>>) -> !firrtl.bundle<a: uint<1>>
    %3 = firrtl.subfield %2(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
    %4 = firrtl.subfield %ax(0) : (!firrtl.bundle<a: bundle<a: uint<1>>>) -> !firrtl.bundle<a: uint<1>>
    %5 = firrtl.subfield %4(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %3, %5 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.partialconnect %3, %5 : !firrtl.uint<1>, !firrtl.uint<1>
    // a.a.a <= ax.a.a
    // CHECK: firrtl.connect %a_a_a, %ax_a_a
    // a.a.a <- ax.a.a
    // CHECK-NEXT: firrtl.connect %a_a_a, %ax_a_a
    %b = firrtl.wire  : !firrtl.bundle<a: bundle<a flip: uint<1>>>
    %bx = firrtl.wire  : !firrtl.bundle<a: bundle<a flip: uint<1>>>
    firrtl.connect %b, %bx : !firrtl.bundle<a: bundle<a flip: uint<1>>>, !firrtl.bundle<a: bundle<a flip: uint<1>>>
    firrtl.partialconnect %b, %bx : !firrtl.bundle<a: bundle<a flip: uint<1>>>, !firrtl.bundle<a: bundle<a flip: uint<1>>>
    // b <= bx
    // CHECK: firrtl.connect %bx_a_a, %b_a_a
    // b <- bx
    // CHECK: firrtl.connect %bx_a_a, %b_a_a
    %6 = firrtl.subfield %b(0) : (!firrtl.bundle<a: bundle<a flip: uint<1>>>) -> !firrtl.bundle<a flip: uint<1>>
    %7 = firrtl.subfield %bx(0) : (!firrtl.bundle<a: bundle<a flip: uint<1>>>) -> !firrtl.bundle<a flip: uint<1>>
    firrtl.connect %6, %7 : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>
    firrtl.partialconnect %6, %7 : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>
    // b.a <= bx.a
    // CHECK: firrtl.connect %bx_a_a, %b_a_a
    // b.a <- bx.a
    // CHECK: firrtl.connect %bx_a_a, %b_a_a
    %8 = firrtl.subfield %b(0) : (!firrtl.bundle<a: bundle<a flip: uint<1>>>) -> !firrtl.bundle<a flip: uint<1>>
    %9 = firrtl.subfield %8(0) : (!firrtl.bundle<a flip: uint<1>>) -> !firrtl.uint<1>
    %10 = firrtl.subfield %bx(0) : (!firrtl.bundle<a: bundle<a flip: uint<1>>>) -> !firrtl.bundle<a flip: uint<1>>
    %11 = firrtl.subfield %10(0) : (!firrtl.bundle<a flip: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %9, %11 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.partialconnect %9, %11 : !firrtl.uint<1>, !firrtl.uint<1>
    // b.a.a <= bx.a.a
    // CHECK: firrtl.connect %b_a_a, %bx_a_a
    // b.a.a <- bx.a.a
    // CHECK: firrtl.connect %b_a_a, %bx_a_a
    %c = firrtl.wire  : !firrtl.bundle<a flip: bundle<a: uint<1>>>
    %cx = firrtl.wire  : !firrtl.bundle<a flip: bundle<a: uint<1>>>
    firrtl.connect %c, %cx : !firrtl.bundle<a flip: bundle<a: uint<1>>>, !firrtl.bundle<a flip: bundle<a: uint<1>>>
    firrtl.partialconnect %c, %cx : !firrtl.bundle<a flip: bundle<a: uint<1>>>, !firrtl.bundle<a flip: bundle<a: uint<1>>>
    // c <= cx
    // CHECK: firrtl.connect %cx_a_a, %c_a_a
    // c <- cx
    // CHECK: firrtl.connect %cx_a_a, %c_a_a
    %12 = firrtl.subfield %c(0) : (!firrtl.bundle<a flip: bundle<a: uint<1>>>) -> !firrtl.bundle<a: uint<1>>
    %13 = firrtl.subfield %cx(0) : (!firrtl.bundle<a flip: bundle<a: uint<1>>>) -> !firrtl.bundle<a: uint<1>>
    firrtl.connect %12, %13 : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    firrtl.partialconnect %12, %13 : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    // c.a <= cx.a
    // CHECK: firrtl.connect %c_a_a, %cx_a_a
    // c.a <- cx.a
    // CHECK: firrtl.connect %c_a_a, %cx_a_a
    %14 = firrtl.subfield %c(0) : (!firrtl.bundle<a flip: bundle<a: uint<1>>>) -> !firrtl.bundle<a: uint<1>>
    %15 = firrtl.subfield %14(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
    %16 = firrtl.subfield %cx(0) : (!firrtl.bundle<a flip: bundle<a: uint<1>>>) -> !firrtl.bundle<a: uint<1>>
    %17 = firrtl.subfield %16(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %15, %17 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.partialconnect %15, %17 : !firrtl.uint<1>, !firrtl.uint<1>
    // c.a.a <= cx.a.a
    // CHECK: firrtl.connect %c_a_a, %cx_a_a
    // c.a.a <- cx.a.a
    // CHECK: firrtl.connect %c_a_a, %cx_a_a
    %d = firrtl.wire  : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    %dx = firrtl.wire  : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    firrtl.connect %d, %dx : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>, !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    firrtl.partialconnect %d, %dx : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>, !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    // d <= dx
    // CHECK: firrtl.connect %d_a_a, %dx_a_a
    // d <- dx
    // CHECK: firrtl.connect %d_a_a, %dx_a_a
    %18 = firrtl.subfield %d(0) : (!firrtl.bundle<a flip: bundle<a flip: uint<1>>>) -> !firrtl.bundle<a flip: uint<1>>
    %19 = firrtl.subfield %dx(0) : (!firrtl.bundle<a flip: bundle<a flip: uint<1>>>) -> !firrtl.bundle<a flip: uint<1>>
    firrtl.connect %18, %19 : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>
    firrtl.partialconnect %18, %19 : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>
    // d.a <= dx.a
    // CHECK: firrtl.connect %dx_a_a, %d_a_a
    // d.a <- dx.a
    // CHECK: firrtl.connect %dx_a_a, %d_a_a
    %20 = firrtl.subfield %d(0) : (!firrtl.bundle<a flip: bundle<a flip: uint<1>>>) -> !firrtl.bundle<a flip: uint<1>>
    %21 = firrtl.subfield %20(0) : (!firrtl.bundle<a flip: uint<1>>) -> !firrtl.uint<1>
    %22 = firrtl.subfield %dx(0) : (!firrtl.bundle<a flip: bundle<a flip: uint<1>>>) -> !firrtl.bundle<a flip: uint<1>>
    %23 = firrtl.subfield %22(0) : (!firrtl.bundle<a flip: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %21, %23 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.partialconnect %21, %23 : !firrtl.uint<1>, !firrtl.uint<1>
    // d.a.a <= dx.a.a
    // CHECK: firrtl.connect %d_a_a, %dx_a_a
    // d.a.a <- dx.a.a
    // CHECK: firrtl.connect %d_a_a, %dx_a_a
  }

// Test corner cases of partial connect semantics.
 // CHECK-LABEL: firrtl.module @PartialConnectEdgeCases
  firrtl.module @PartialConnectEdgeCases() {
    // Only matching fields are connected.
    %a = firrtl.wire : !firrtl.bundle<a: uint<1>, b: uint<1>, c: uint<1>>
    %b = firrtl.wire : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.partialconnect %a, %b : !firrtl.bundle<a: uint<1>, b: uint<1>, c: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
    // CHECK: firrtl.connect %a_a, %b_a
    // CHECK-NEXT: firrtl.connect %a_b, %b_b
    // CHECK-NOT: firrtl.connect %a_

    firrtl.partialconnect %b, %a : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>, c: uint<1>>
    // CHECK: firrtl.connect %b_a, %a_a
    // CHECK-NEXT: firrtl.connect %b_b, %a_b
    // CHECK-NOT: firrtl.connect %b_

    // Only the first 'n' elements in a vector are connected.
    %c = firrtl.wire : !firrtl.vector<uint<1>, 2>
    %d = firrtl.wire : !firrtl.vector<uint<1>, 3>
    firrtl.partialconnect %c, %d : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 3>
    // CHECK: firrtl.connect %c_0, %d_0
    // CHECK-NEXT: firrtl.connect %c_1, %d_1
    // CHECK-NOT: firrtl.connect %c_

    firrtl.partialconnect %d, %c : !firrtl.vector<uint<1>, 3>, !firrtl.vector<uint<1>, 2>
    // CHECK: firrtl.connect %d_0, %c_0
    // CHECK-NEXT: firrtl.connect %d_1, %c_1
    // CHECK-NOT: firrtl.connect %d_
  }


// Test partial connect truncation.
 // CHECK-LABEL: firrtl.module @PartialConnectTruncation
  firrtl.module @PartialConnectTruncation() {
    // It should not truncate when they are the same
    %a = firrtl.wire : !firrtl.uint<0>
    %b = firrtl.wire : !firrtl.uint<0>
    firrtl.partialconnect %a, %b : !firrtl.uint<0>, !firrtl.uint<0>
    // CHECK: firrtl.connect %a, %b : !firrtl.uint<0>, !firrtl.uint<0>

    // It should truncate the larger source.
    %c = firrtl.wire : !firrtl.uint<2>
    %d = firrtl.wire : !firrtl.uint<3>
    firrtl.partialconnect %c, %d : !firrtl.uint<2>, !firrtl.uint<3>
    // CHECK: [[TAIL:%.*]] = firrtl.tail %d, 1 : (!firrtl.uint<3>) -> !firrtl.uint<2>
    // CHECK: firrtl.connect %c, [[TAIL]] : !firrtl.uint<2>, !firrtl.uint<2>

    // It should truncate and cast the larger source.
    %e = firrtl.wire : !firrtl.sint<2>
    %f = firrtl.wire : !firrtl.sint<3>
    firrtl.partialconnect %e, %f : !firrtl.sint<2>, !firrtl.sint<3>
    // CHECK: [[TAIL:%.*]] = firrtl.tail %f, 1 : (!firrtl.sint<3>) -> !firrtl.uint<2>
    // CHECK: [[CAST:%.*]] = firrtl.asSInt [[TAIL]] : (!firrtl.uint<2>) -> !firrtl.sint<2>
    // CHECK: firrtl.connect %e, [[CAST]] : !firrtl.sint<2>, !firrtl.sint<2>
  }

// Bug: must strip the flip type from the LHS of a value.
 // CHECK-LABEL: firrtl.module @PartialConnectLHSFlip
  firrtl.module @PartialConnectLHSFlip(in %a: !firrtl.bundle<b: bundle<c flip: uint<2>>>) { }
   // CHECK-LABEL: firrtl.module @FooFlipType
  firrtl.module @FooFlipType(in %a: !firrtl.bundle<b: bundle<c flip: uint<2>>>) {
    %mgmt_a = firrtl.instance mgmt @PartialConnectLHSFlip(in a: !firrtl.bundle<b: bundle<c flip: uint<2>>>)
    %0 = firrtl.subfield %mgmt_a(0) : (!firrtl.bundle<b: bundle<c flip: uint<2>>>) -> !firrtl.bundle<c flip: uint<2>>
    %1 = firrtl.subfield %0(0) : (!firrtl.bundle<c flip: uint<2>>) -> !firrtl.uint<2>
    // CHECK: firrtl.connect %a_b_c, %mgmt_a_b_c : !firrtl.uint<2>, !firrtl.uint<2>
    firrtl.partialconnect %mgmt_a, %a : !firrtl.bundle<b: bundle<c flip: uint<2>>>, !firrtl.bundle<b: bundle<c flip: uint<2>>>
  }

// Test partial connect with analogs are transformed to attaches.
  firrtl.module @PartialConnectAnalogs() {
    // CHECK-LABEL: @PartialConnectAnalogs
    %a = firrtl.wire : !firrtl.bundle<a: analog<1>>
    %b = firrtl.wire : !firrtl.bundle<a: analog<1>>
    firrtl.partialconnect %a, %b : !firrtl.bundle<a: analog<1>>, !firrtl.bundle<a: analog<1>>
    // CHECK: firrtl.attach %a_a, %b_a : !firrtl.analog<1>, !firrtl.analog<1>
  }

// Test that a vector of bundles with a write works.
 // CHECK-LABEL: firrtl.module @aofs
    firrtl.module @aofs(in %a: !firrtl.uint<1>, in %sel: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>>, 4>) {
      %0 = firrtl.subindex %b[0] : !firrtl.vector<bundle<wo: uint<1>>, 4>
      %1 = firrtl.subfield %0(0) : (!firrtl.bundle<wo: uint<1>>) -> !firrtl.uint<1>
      %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
      firrtl.connect %1, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
      %2 = firrtl.subindex %b[1] : !firrtl.vector<bundle<wo: uint<1>>, 4>
      %3 = firrtl.subfield %2(0) : (!firrtl.bundle<wo: uint<1>>) -> !firrtl.uint<1>
      %invalid_ui1_0 = firrtl.invalidvalue : !firrtl.uint<1>
      firrtl.connect %3, %invalid_ui1_0 : !firrtl.uint<1>, !firrtl.uint<1>
      %4 = firrtl.subindex %b[2] : !firrtl.vector<bundle<wo: uint<1>>, 4>
      %5 = firrtl.subfield %4(0) : (!firrtl.bundle<wo: uint<1>>) -> !firrtl.uint<1>
      %invalid_ui1_1 = firrtl.invalidvalue : !firrtl.uint<1>
      firrtl.connect %5, %invalid_ui1_1 : !firrtl.uint<1>, !firrtl.uint<1>
      %6 = firrtl.subindex %b[3] : !firrtl.vector<bundle<wo: uint<1>>, 4>
      %7 = firrtl.subfield %6(0) : (!firrtl.bundle<wo: uint<1>>) -> !firrtl.uint<1>
      %invalid_ui1_2 = firrtl.invalidvalue : !firrtl.uint<1>
      firrtl.connect %7, %invalid_ui1_2 : !firrtl.uint<1>, !firrtl.uint<1>
      %8 = firrtl.subaccess %b[%sel] : !firrtl.vector<bundle<wo: uint<1>>, 4>, !firrtl.uint<2>
      %9 = firrtl.subfield %8(0) : (!firrtl.bundle<wo: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %9, %a : !firrtl.uint<1>, !firrtl.uint<1>
    }


// Test that annotations on aggregate ports are copied.
  firrtl.extmodule @Sub1(in a: !firrtl.vector<uint<1>, 2> [{a}])
  // CHECK-LABEL: firrtl.extmodule @Sub1
  // CHECK-COUNT-2: [{b}]
  // CHECK-NOT: [{a}]
  firrtl.module @Port(in %a: !firrtl.vector<uint<1>, 2> [{b}]) {
    %sub_a = firrtl.instance sub @Sub1(in a: !firrtl.vector<uint<1>, 2>)
    firrtl.connect %sub_a, %a : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }

// Test that annotations on subfield/subindices of ports are only applied to
// matching fieldIDs.
    // The annotation should be copied to just a.a.  The firrtl.hello arg
    // attribute should be copied to each new port.
    firrtl.module @PortBundle(in %a: !firrtl.bundle<a: uint<1>, b flip: uint<1>> [#firrtl.subAnno<fieldID = 1, {a}>]) {}
    // CHECK-LABEL: firrtl.module @PortBundle
    // CHECK-COUNT-1: [{a}]

// Test that a truncating connect emitted during lower types correctly adds an
// AsPassive cast on a FlipType originating from an instance.
//
// See: https://github.com/llvm/circt/issues/1276

  firrtl.extmodule @Bar2(in a: !firrtl.uint<2>)
// CHECK-LABEL: firrtl.module @TruncatingConnectWithFlip
  firrtl.module @TruncatingConnectWithFlip() {
    // CHECK: %[[a_b:.+]] = firrtl.wire
    %a = firrtl.wire  : !firrtl.bundle<b: uint<1>>
    // CHECK-NEXT: %bar_a = firrtl.instance bar @Bar
    %bar_a = firrtl.instance bar @Bar2(in a: !firrtl.uint<2>)
    // CHECK-NEXT: %invalid_ui2 = firrtl.invalidvalue
    %invalid_ui2 = firrtl.invalidvalue : !firrtl.uint<2>
    // CHECK-NEXT: firrtl.connect %bar_a, %invalid_ui2
    firrtl.connect %bar_a, %invalid_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    // CHECK: %[[bar_a_tail:.+]] = firrtl.tail %bar_a, 1
    %0 = firrtl.subfield %a(0) : (!firrtl.bundle<b: uint<1>>) -> !firrtl.uint<1>
    // CHECK-NEXT: firrtl.connect %[[a_b]], %[[bar_a_tail]]
    firrtl.partialconnect %0, %bar_a : !firrtl.uint<1>, !firrtl.uint<2>
  }


// circuit Foo:
//   module Foo:
//     input a: UInt<2>[2][2]
//     input sel: UInt<2>
//     output b: UInt<2>
//
//     b <= a[sel][sel]

  firrtl.module @multidimRead(in %a: !firrtl.vector<vector<uint<2>, 2>, 2>, in %sel: !firrtl.uint<2>, out %b: !firrtl.uint<2>) {
    %0 = firrtl.subaccess %a[%sel] : !firrtl.vector<vector<uint<2>, 2>, 2>, !firrtl.uint<2>
    %1 = firrtl.subaccess %0[%sel] : !firrtl.vector<uint<2>, 2>, !firrtl.uint<2>
    firrtl.connect %b, %1 : !firrtl.uint<2>, !firrtl.uint<2>
  }

// CHECK-LABEL: firrtl.module @multidimRead(in %a_0_0: !firrtl.uint<2>, in %a_0_1: !firrtl.uint<2>, in %a_1_0: !firrtl.uint<2>, in %a_1_1: !firrtl.uint<2>, in %sel: !firrtl.uint<2>, out %b: !firrtl.uint<2>) {
// CHECK-NEXT:      %0 = firrtl.multibit_mux %sel, %a_1_0, %a_0_0 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT:      %1 = firrtl.multibit_mux %sel, %a_1_1, %a_0_1 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT:      %2 = firrtl.multibit_mux %sel, %1, %0 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT:      firrtl.connect %b, %2 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }

//  module Foo:
//    input b: UInt<1>
//    input sel: UInt<2>
//    input default: UInt<1>[4]
//    output a: UInt<1>[4]
//
//     a <= default
//     a[sel] <= b

  firrtl.module @write1D(in %b: !firrtl.uint<1>, in %sel: !firrtl.uint<2>, in %default: !firrtl.vector<uint<1>, 2>, out %a: !firrtl.vector<uint<1>, 2>) {
    firrtl.connect %a, %default : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subaccess %a[%sel] : !firrtl.vector<uint<1>, 2>, !firrtl.uint<2>
    firrtl.connect %0, %b : !firrtl.uint<1>, !firrtl.uint<1>
  }
// CHECK-LABEL:    firrtl.module @write1D(in %b: !firrtl.uint<1>, in %sel: !firrtl.uint<2>, in %default_0: !firrtl.uint<1>, in %default_1: !firrtl.uint<1>, out %a_0: !firrtl.uint<1>, out %a_1: !firrtl.uint<1>) {
// CHECK-NEXT:      firrtl.connect %a_0, %default_0 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT:      firrtl.connect %a_1, %default_1 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT:      %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
// CHECK-NEXT:      %0 = firrtl.eq %sel, %c0_ui1 : (!firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:      firrtl.when %0  {
// CHECK-NEXT:        firrtl.connect %a_0, %b : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
// CHECK-NEXT:      %1 = firrtl.eq %sel, %c1_ui1 : (!firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:      firrtl.when %1  {
// CHECK-NEXT:        firrtl.connect %a_1, %b : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT:      }
// CHECK-NEXT:    }


// circuit Foo:
//   module Foo:
//     input sel: UInt<1>
//     input b: UInt<2>
//     output a: UInt<2>[2][2]
//
//     a[sel][sel] <= b

  firrtl.module @multidimWrite(in %sel: !firrtl.uint<1>, in %b: !firrtl.uint<2>, out %a: !firrtl.vector<vector<uint<2>, 2>, 2>) {
    %0 = firrtl.subaccess %a[%sel] : !firrtl.vector<vector<uint<2>, 2>, 2>, !firrtl.uint<1>
    %1 = firrtl.subaccess %0[%sel] : !firrtl.vector<uint<2>, 2>, !firrtl.uint<1>
    firrtl.connect %1, %b : !firrtl.uint<2>, !firrtl.uint<2>
  }
// CHECK-LABEL:    firrtl.module @multidimWrite(in %sel: !firrtl.uint<1>, in %b: !firrtl.uint<2>, out %a_0_0: !firrtl.uint<2>, out %a_0_1: !firrtl.uint<2>, out %a_1_0: !firrtl.uint<2>, out %a_1_1: !firrtl.uint<2>) {
// CHECK-NEXT:      %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
// CHECK-NEXT:      %0 = firrtl.eq %sel, %c0_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:      firrtl.when %0  {
// CHECK-NEXT:        %c0_ui1_0 = firrtl.constant 0 : !firrtl.uint<1>
// CHECK-NEXT:        %2 = firrtl.eq %sel, %c0_ui1_0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:        firrtl.when %2  {
// CHECK-NEXT:          firrtl.connect %a_0_0, %b : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT:        }
// CHECK-NEXT:        %c1_ui1_1 = firrtl.constant 1 : !firrtl.uint<1>
// CHECK-NEXT:        %3 = firrtl.eq %sel, %c1_ui1_1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:        firrtl.when %3  {
// CHECK-NEXT:          firrtl.connect %a_0_1, %b : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
// CHECK-NEXT:      %1 = firrtl.eq %sel, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:      firrtl.when %1  {
// CHECK-NEXT:        %c0_ui1_0 = firrtl.constant 0 : !firrtl.uint<1>
// CHECK-NEXT:        %2 = firrtl.eq %sel, %c0_ui1_0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:        firrtl.when %2  {
// CHECK-NEXT:          firrtl.connect %a_1_0, %b : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT:        }
// CHECK-NEXT:        %c1_ui1_1 = firrtl.constant 1 : !firrtl.uint<1>
// CHECK-NEXT:        %3 = firrtl.eq %sel, %c1_ui1_1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:        firrtl.when %3  {
// CHECK-NEXT:          firrtl.connect %a_1_1, %b : !firrtl.uint<2>, !firrtl.uint<2>
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
  firrtl.module @writeVectorOfBundle1D(in %a: !firrtl.bundle<wo: uint<1>, valid: uint<2>>, in %def: !firrtl.vector<bundle<wo: uint<1>, valid: uint<2>>, 2>, in %sel: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>, valid: uint<2>>, 2>) {
    firrtl.connect %b, %def : !firrtl.vector<bundle<wo: uint<1>, valid: uint<2>>, 2>, !firrtl.vector<bundle<wo: uint<1>, valid: uint<2>>, 2>
    %0 = firrtl.subaccess %b[%sel] : !firrtl.vector<bundle<wo: uint<1>, valid: uint<2>>, 2>, !firrtl.uint<2>
    %1 = firrtl.subfield %0(0) : (!firrtl.bundle<wo: uint<1>, valid: uint<2>>) -> !firrtl.uint<1>
    %2 = firrtl.subfield %a(0) : (!firrtl.bundle<wo: uint<1>, valid: uint<2>>) -> !firrtl.uint<1>
    firrtl.connect %1, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }

// CHECK-LABEL:    firrtl.module @writeVectorOfBundle1D(in %a_wo: !firrtl.uint<1>, in %a_valid: !firrtl.uint<2>, in %def_0_wo: !firrtl.uint<1>, in %def_0_valid: !firrtl.uint<2>, in %def_1_wo: !firrtl.uint<1>, in %def_1_valid: !firrtl.uint<2>, in %sel: !firrtl.uint<2>, out %b_0_wo: !firrtl.uint<1>, out %b_0_valid: !firrtl.uint<2>, out %b_1_wo: !firrtl.uint<1>, out %b_1_valid: !firrtl.uint<2>) {
// CHECK-NEXT:      firrtl.connect %b_0_wo, %def_0_wo : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT:      firrtl.connect %b_0_valid, %def_0_valid : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT:      firrtl.connect %b_1_wo, %def_1_wo : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT:      firrtl.connect %b_1_valid, %def_1_valid : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT:      %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
// CHECK-NEXT:      %0 = firrtl.eq %sel, %c0_ui1 : (!firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:      firrtl.when %0  {
// CHECK-NEXT:        firrtl.connect %b_0_wo, %a_wo : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
// CHECK-NEXT:      %1 = firrtl.eq %sel, %c1_ui1 : (!firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:      firrtl.when %1  {
// CHECK-NEXT:        firrtl.connect %b_1_wo, %a_wo : !firrtl.uint<1>, !firrtl.uint<1>
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
  firrtl.module @multiSubaccess(in %a: !firrtl.vector<vector<uint<2>, 2>, 2>, in %sel1: !firrtl.uint<1>, in %sel2: !firrtl.uint<1>, out %b: !firrtl.uint<2>, out %c: !firrtl.uint<2>) {
    %0 = firrtl.subaccess %a[%sel1] : !firrtl.vector<vector<uint<2>, 2>, 2>, !firrtl.uint<1>
    %1 = firrtl.subaccess %0[%sel1] : !firrtl.vector<uint<2>, 2>, !firrtl.uint<1>
    firrtl.connect %b, %1 : !firrtl.uint<2>, !firrtl.uint<2>
    %2 = firrtl.subaccess %a[%sel1] : !firrtl.vector<vector<uint<2>, 2>, 2>, !firrtl.uint<1>
    %3 = firrtl.subaccess %2[%sel2] : !firrtl.vector<uint<2>, 2>, !firrtl.uint<1>
    firrtl.connect %c, %3 : !firrtl.uint<2>, !firrtl.uint<2>
  }

// CHECK-LABEL:    firrtl.module @multiSubaccess(in %a_0_0: !firrtl.uint<2>, in %a_0_1: !firrtl.uint<2>, in %a_1_0: !firrtl.uint<2>, in %a_1_1: !firrtl.uint<2>, in %sel1: !firrtl.uint<1>, in %sel2: !firrtl.uint<1>, out %b: !firrtl.uint<2>, out %c: !firrtl.uint<2>) {
// CHECK-NEXT:      %0 = firrtl.multibit_mux %sel1, %a_1_0, %a_0_0 : !firrtl.uint<1>, !firrtl.uint<2>
// CHECK-NEXT:      %1 = firrtl.multibit_mux %sel1, %a_1_1, %a_0_1 : !firrtl.uint<1>, !firrtl.uint<2>
// CHECK-NEXT:      %2 = firrtl.multibit_mux %sel1, %1, %0 : !firrtl.uint<1>, !firrtl.uint<2>
// CHECK-NEXT:      firrtl.connect %b, %2 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT:      %3 = firrtl.multibit_mux %sel1, %a_1_0, %a_0_0 : !firrtl.uint<1>, !firrtl.uint<2>
// CHECK-NEXT:      %4 = firrtl.multibit_mux %sel1, %a_1_1, %a_0_1 : !firrtl.uint<1>, !firrtl.uint<2>
// CHECK-NEXT:      %5 = firrtl.multibit_mux %sel2, %4, %3 : !firrtl.uint<1>, !firrtl.uint<2>
// CHECK-NEXT:      firrtl.connect %c, %5 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT:    }


// Handle zero-length vector subaccess
  // CHECK-LABEL: zvec
  firrtl.module @zvec(in %i: !firrtl.vector<bundle<a: uint<8>, b: uint<4>>, 0>, in %sel: !firrtl.uint<1>, out %foo: !firrtl.vector<uint<1>, 0>, out %o: !firrtl.uint<8>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.subaccess %foo[%c0_ui1] : !firrtl.vector<uint<1>, 0>, !firrtl.uint<1>
    firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = firrtl.subaccess %i[%sel] : !firrtl.vector<bundle<a: uint<8>, b: uint<4>>, 0>, !firrtl.uint<1>
    %2 = firrtl.subfield %1(0) : (!firrtl.bundle<a: uint<8>, b: uint<4>>) -> !firrtl.uint<8>
    firrtl.connect %o, %2 : !firrtl.uint<8>, !firrtl.uint<8>
  // CHECK: firrtl.connect %o, %invalid_ui8
  }

// Test InstanceOp with port annotations.
// CHECK-LABEL firrtl.module @Bar3
  firrtl.module @Bar3(in %a: !firrtl.uint<1>, out %b: !firrtl.bundle<baz: uint<1>, qux: uint<1>>) {
  }
  // CHECK-LABEL firrtl.module @Foo3
  firrtl.module @Foo3() {
    // CHECK: in a: !firrtl.uint<1> [{one}], out b_baz: !firrtl.uint<1> [{two}], out b_qux: !firrtl.uint<1>
    %bar_a, %bar_b = firrtl.instance bar @Bar3(in a: !firrtl.uint<1> [{one}], out b: !firrtl.bundle<baz: uint<1>, qux: uint<1>> [#firrtl.subAnno<fieldID = 1, {two}>])
  }


// Test MemOp with port annotations.
// circuit Foo: %[[{"a":null,"target":"~Foo|Foo>bar.r"},
//                 {"b":null,"target":"~Foo|Foo>bar.r.data"},
//                 {"c":null,"target":"~Foo|Foo>bar.w.en"},
//                 {"d":null,"target":"~Foo|Foo>bar.w.data.qux"},
//                 {"e":null,"target":"~Foo|Foo>bar.rw.wmode"}
//                 {"f":null,"target":"~Foo|Foo>bar.rw.wmask.baz"}]]

// CHECK-LABEL: firrtl.module @Foo4
// FLATTEN-LABEL: firrtl.module @Foo4
  firrtl.module @Foo4() {
    // CHECK: firrtl.mem
    // CHECK-SAME: portAnnotations = [
    // CHECK-SAME: [{a}, #firrtl.subAnno<fieldID = 4, {b}>],
    // CHECK-SAME: [#firrtl.subAnno<fieldID = 2, {c}>]
    // CHECK-SAME: [#firrtl.subAnno<fieldID = 4, {e}>, #firrtl.subAnno<fieldID = 7, {f}>]

    // CHECK: firrtl.mem
    // CHECK-SAME: portAnnotations = [
    // CHECK-SAME: [{a}, #firrtl.subAnno<fieldID = 4, {b}>],
    // CHECK-SAME: [#firrtl.subAnno<fieldID = 2, {c}>, #firrtl.subAnno<fieldID = 4, {d}>]
    // CHECK-SAME: [#firrtl.subAnno<fieldID = 4, {e}>]

    // FLATTEN: firrtl.mem
    // FLATTEN-SAME: portAnnotations = [
    // FLATTEN-SAME: [{a}, #firrtl.subAnno<fieldID = 4, {b}>],
    // FLATTEN-SAME: [#firrtl.subAnno<fieldID = 2, {c}>]
    // FLATTEN-SAME: [#firrtl.subAnno<fieldID = 4, {e}>, #firrtl.subAnno<fieldID = 7, {f}>]

    // FLATTEN: firrtl.mem
    // FLATTEN-SAME: portAnnotations = [
    // FLATTEN-SAME: [{a}, #firrtl.subAnno<fieldID = 4, {b}>],
    // FLATTEN-SAME: [#firrtl.subAnno<fieldID = 2, {c}>, #firrtl.subAnno<fieldID = 4, {d}>]
    // FLATTEN-SAME: [#firrtl.subAnno<fieldID = 4, {e}>]
    %bar_r, %bar_w, %bar_rw = firrtl.mem Undefined  {depth = 16 : i64, name = "bar",
        portAnnotations = [
          [{a}, #firrtl.subAnno<fieldID = 4, {b}>],
          [#firrtl.subAnno<fieldID = 2, {c}>, #firrtl.subAnno<fieldID = 6, {d}>],
          [#firrtl.subAnno<fieldID = 4, {e}>, #firrtl.subAnno<fieldID = 12, {f}>]
        ],
        portNames = ["r", "w", "rw"], readLatency = 0 : i32, writeLatency = 1 : i32} :
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<baz: uint<8>, qux: uint<8>>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<baz: uint<8>, qux: uint<8>>, mask: bundle<baz: uint<1>, qux: uint<1>>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<baz: uint<8>, qux: uint<8>>, wmode: uint<1>, wdata: bundle<baz: uint<8>, qux: uint<8>>, wmask: bundle<baz: uint<1>, qux: uint<1>>>
  }

// Test that partial connects can extend
// CHECK-LABEL: firrtl.module @TLBBB
firrtl.module @TLBBB() {
  %invalid_ui9 = firrtl.invalidvalue : !firrtl.uint<9>
  %in_bits = firrtl.wire  : !firrtl.uint<9>
  firrtl.connect %in_bits, %invalid_ui9 : !firrtl.uint<9>, !firrtl.uint<9>
  %out0_bits = firrtl.wire  : !firrtl.uint<128>
// CHECK: firrtl.connect %out0_bits
  firrtl.partialconnect %out0_bits, %in_bits : !firrtl.uint<128>, !firrtl.uint<9>
}

// This simply has to not crash
// CHECK-LABEL: firrtl.module @vecmem
// FLATTEN-LABEL: firrtl.module @vecmem
firrtl.module @vecmem(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
  %vmem_MPORT, %vmem_rdwrPort = firrtl.mem Undefined  {depth = 32 : i64, name = "vmem", portNames = ["MPORT", "rdwrPort"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data flip: vector<uint<17>, 8>>, !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, rdata flip: vector<uint<17>, 8>, wmode: uint<1>, wdata: vector<uint<17>, 8>, wmask: vector<uint<1>, 8>>
}

// CHECK-LABEL: firrtl.module @bofa
firrtl.module @bofa(out %auto: !firrtl.bundle<io_out: bundle<foo: bundle<bar: analog<1>>>>) {
  %0 = firrtl.subfield %auto(0) : (!firrtl.bundle<io_out: bundle<foo: bundle<bar: analog<1>>>>) -> !firrtl.bundle<foo: bundle<bar: analog<1>>>
  %io = firrtl.wire  : !firrtl.bundle<foo: bundle<bar: analog<1>>>
  firrtl.partialconnect %0, %io : !firrtl.bundle<foo: bundle<bar: analog<1>>>, !firrtl.bundle<foo: bundle<bar: analog<1>>>
// CHECK: firrtl.attach %auto_io_out_foo_bar, %io_foo_bar
}

// Issue 1436
firrtl.extmodule @is1436_BAR(out io: !firrtl.bundle<llWakeup flip: vector<uint<1>, 1>>)
// CHECK-LABEL: firrtl.module @is1436_FOO
firrtl.module @is1436_FOO() {
  %thing_io = firrtl.instance thing @is1436_BAR(out io: !firrtl.bundle<llWakeup flip: vector<uint<1>, 1>>)
  %0 = firrtl.subfield %thing_io(0) : (!firrtl.bundle<llWakeup flip: vector<uint<1>, 1>>) -> !firrtl.vector<uint<1>, 1>
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %1 = firrtl.subaccess %0[%c0_ui2] : !firrtl.vector<uint<1>, 1>, !firrtl.uint<2>
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  firrtl.connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module @BitCast1
  firrtl.module @BitCast1(out %o_0: !firrtl.uint<1>, out %o_1: !firrtl.uint<1>) {
    %a1 = firrtl.wire : !firrtl.uint<4>
    %b = firrtl.bitcast %a1 : (!firrtl.uint<4>) -> (!firrtl.vector<uint<2>, 2>)
    // CHECK:  %[[v0:.+]] = firrtl.bits %a1 1 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<2>
    // CHECK-NEXT:  %[[v1:.+]] = firrtl.bits %a1 3 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<2>
    // CHECK-NEXT:  %[[v2:.+]] = firrtl.cat %[[v1]], %[[v0]] : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<4>
    %c = firrtl.bitcast %b : (!firrtl.vector<uint<2>, 2>) -> (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<2>>)
    // CHECK-NEXT:  %[[v3:.+]] = firrtl.bits %[[v2]] 0 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT:  %[[v4:.+]] = firrtl.bits %[[v2]] 1 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT:  %[[v5:.+]] = firrtl.bits %[[v2]] 3 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<2>
    %d = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<2>>
    // CHECK-NEXT:  %d_valid = firrtl.wire  : !firrtl.uint<1>
    // CHECK-NEXT:  %d_ready = firrtl.wire  : !firrtl.uint<1>
    // CHECK-NEXT:  %d_data = firrtl.wire  : !firrtl.uint<2>
    firrtl.connect %d , %c: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<2>>, !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<2>>
    // CHECK-NEXT:  firrtl.connect %d_valid, %[[v3]] : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT:  firrtl.connect %d_ready, %[[v4]] : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT:  firrtl.connect %d_data, %[[v5]] : !firrtl.uint<2>, !firrtl.uint<2>
    %e = firrtl.bitcast %d : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<2>>) -> (!firrtl.bundle<addr: uint<2>, data : vector<uint<1>, 2>>)
    // CHECK-NEXT:  %[[v6:.+]] = firrtl.cat %d_ready, %d_valid : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    // CHECK-NEXT:  %[[v7:.+]] = firrtl.cat %d_data, %[[v6]] : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<4>
    // CHECK-NEXT:  %[[v8:.+]] = firrtl.bits %[[v7]] 1 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<2>
    // CHECK-NEXT:  %[[v9:.+]] = firrtl.bits %[[v7]] 3 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  %o1 = firrtl.subfield %e(1) : (!firrtl.bundle<addr: uint<2>, data : vector<uint<1>, 2>>) -> !firrtl.vector<uint<1>,2>
   %c2 = firrtl.bitcast %a1 : (!firrtl.uint<4>) -> (!firrtl.bundle<valid: bundle<re: bundle<a: uint<1>>, aa: uint<1>>, ready: uint<1>, data: uint<1>>)
    %d2 = firrtl.wire : !firrtl.bundle<valid: bundle<re: bundle<a: uint<1>>, aa: uint<1>>, ready: uint<1>, data: uint<1>>
    firrtl.connect %d2 , %c2: !firrtl.bundle<valid: bundle<re: bundle<a: uint<1>>, aa: uint<1>>, ready: uint<1>, data:
    uint<1>>, !firrtl.bundle<valid: bundle<re: bundle<a: uint<1>>, aa: uint<1>>, ready: uint<1>, data: uint<1>>
   //CHECK: %[[v10:.+]] = firrtl.bits %[[v9]] 0 to 0 : (!firrtl.uint<2>) -> !firrtl.uint<1>
   //CHECK: %[[v11:.+]] = firrtl.bits %[[v9]] 1 to 1 : (!firrtl.uint<2>) -> !firrtl.uint<1>
   //CHECK: %[[v12:.+]] = firrtl.bits %a1 1 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<2>
   //CHECK: %[[v13:.+]] = firrtl.bits %[[v12]] 0 to 0 : (!firrtl.uint<2>) -> !firrtl.uint<1>
   //CHECK: %[[v14:.+]] = firrtl.bits %[[v13]] 0 to 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
   //CHECK: %[[v15:.+]] = firrtl.bits %[[v12]] 1 to 1 : (!firrtl.uint<2>) -> !firrtl.uint<1>
   //CHECK: %[[v16:.+]] = firrtl.bits %a1 2 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<1>
   //CHECK: %[[v17:.+]] = firrtl.bits %a1 3 to 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
   //CHECK: %[[d2_valid_re_a:.+]] = firrtl.wire  : !firrtl.uint<1>
   //CHECK: %[[d2_valid_aa:.+]] = firrtl.wire  : !firrtl.uint<1>
   //CHECK: %[[d2_ready:.+]] = firrtl.wire  : !firrtl.uint<1>
   //CHECK: %[[d2_data:.+]] = firrtl.wire  : !firrtl.uint<1>
   //CHECK: firrtl.connect %[[d2_valid_re_a]], %[[v14]] : !firrtl.uint<1>, !firrtl.uint<1>
   //CHECK: firrtl.connect %[[d2_valid_aa]], %[[v15]] : !firrtl.uint<1>, !firrtl.uint<1>
   //CHECK: firrtl.connect %[[d2_ready]], %[[v16]] : !firrtl.uint<1>, !firrtl.uint<1>
   //CHECK: firrtl.connect %[[d2_data]], %[[v17]] : !firrtl.uint<1>, !firrtl.uint<1>

  }

  firrtl.module @VecMemFlatten(in %clock: !firrtl.clock, in %rAddr: !firrtl.uint<4>, in %rEn: !firrtl.uint<1>, out %rData: !firrtl.vector<uint<8>, 4>, in %wAddr: !firrtl.uint<4>, in %wEn: !firrtl.uint<1>, in %wMask: !firrtl.vector<uint<1>, 4>, in %wData: !firrtl.vector<uint<8>, 4>) {
    %memory_r, %memory_w = firrtl.mem Undefined  {depth = 16 : i64, name = "memory", portNames = ["r", "w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: vector<uint<8>, 4>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: vector<uint<8>, 4>, mask: vector<uint<1>, 4>>
    %0 = firrtl.subfield %memory_r(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: vector<uint<8>, 4>>) -> !firrtl.clock
    firrtl.connect %0, %clock : !firrtl.clock, !firrtl.clock
    %1 = firrtl.subfield %memory_r(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: vector<uint<8>, 4>>) -> !firrtl.uint<1>
    firrtl.connect %1, %rEn : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.subfield %memory_r(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: vector<uint<8>, 4>>) -> !firrtl.uint<4>
    firrtl.connect %2, %rAddr : !firrtl.uint<4>, !firrtl.uint<4>
    %3 = firrtl.subfield %memory_r(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: vector<uint<8>, 4>>) -> !firrtl.vector<uint<8>, 4>
    firrtl.connect %rData, %3 : !firrtl.vector<uint<8>, 4>, !firrtl.vector<uint<8>, 4>
    %4 = firrtl.subfield %memory_w(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: vector<uint<8>, 4>, mask: vector<uint<1>, 4>>) -> !firrtl.clock
    firrtl.connect %4, %clock : !firrtl.clock, !firrtl.clock
    %5 = firrtl.subfield %memory_w(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: vector<uint<8>, 4>, mask: vector<uint<1>, 4>>) -> !firrtl.uint<1>
    firrtl.connect %5, %wEn : !firrtl.uint<1>, !firrtl.uint<1>
    %6 = firrtl.subfield %memory_w(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: vector<uint<8>, 4>, mask: vector<uint<1>, 4>>) -> !firrtl.uint<4>
    firrtl.connect %6, %wAddr : !firrtl.uint<4>, !firrtl.uint<4>
    %7 = firrtl.subfield %memory_w(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: vector<uint<8>, 4>, mask: vector<uint<1>, 4>>) -> !firrtl.vector<uint<1>, 4>
    firrtl.connect %7, %wMask : !firrtl.vector<uint<1>, 4>, !firrtl.vector<uint<1>, 4>
    %8 = firrtl.subfield %memory_w(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: vector<uint<8>, 4>, mask: vector<uint<1>, 4>>) -> !firrtl.vector<uint<8>, 4>
    firrtl.connect %8, %wData : !firrtl.vector<uint<8>, 4>, !firrtl.vector<uint<8>, 4>
  }
  // FLATTEN-LABEL:  firrtl.module @VecMemFlatten
  // FLATTEN:    %[[memory_r:.+]], %[[memory_w:.+]] = firrtl.mem Undefined  {depth = 16 : i64, name = "memory", portNames = ["r", "w"], readLatency = 0 : i32, writeLatency = 1 : i32}
  // FLATTEN-SAME:    !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<32>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<4>>
  // FLATTEN:    %[[v3:.+]] = firrtl.subfield %[[memory_r]](3)
  // FLATTEN:    %[[v4:.+]] = firrtl.bits %3 7 to 0 : (!firrtl.uint<32>) -> !firrtl.uint<8>
  // FLATTEN:    %[[v5:.+]] = firrtl.bits %3 15 to 8 : (!firrtl.uint<32>) -> !firrtl.uint<8>
  // FLATTEN:    %[[v6:.+]] = firrtl.bits %3 23 to 16 : (!firrtl.uint<32>) -> !firrtl.uint<8>
  // FLATTEN:    %[[v7:.+]] = firrtl.bits %3 31 to 24 : (!firrtl.uint<32>) -> !firrtl.uint<8>
  // FLATTEN:    firrtl.connect %[[memory_r_data_0:.+]], %[[v4]]
  // FLATTEN:    firrtl.connect %[[memory_r_data_1:.+]], %[[v5]]
  // FLATTEN:    firrtl.connect %[[memory_r_data_2:.+]], %[[v6]]
  // FLATTEN:    firrtl.connect %[[memory_r_data_3:.+]], %[[v7]]

  // FLATTEN:    %[[v11:.+]] = firrtl.subfield %[[memory_w]](3)
  // FLATTEN:    %[[v12:.+]] = firrtl.cat %[[memory_w_data_1:.+]], %[[memory_w_data_0:.+]] : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<16>
  // FLATTEN:    %[[v13:.+]] = firrtl.cat %[[memory_w_data_2:.+]], %[[v12]] : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.uint<24>
  // FLATTEN:    %[[v14:.+]] = firrtl.cat %[[memory_w_data_3:.+]], %[[v13]] : (!firrtl.uint<8>, !firrtl.uint<24>) -> !firrtl.uint<32>
  // FLATTEN:    f[[vir:.+]]rtl.connect %[[v11]], %[[v14]] : !firrtl.uint<32>, !firrtl.uint<32>
  // FLATTEN:    %[[v15:.+]] = firrtl.subfield %[[memory_w]](4)
  // FLATTEN:    %[[v16:.+]] = firrtl.cat %[[memory_w_mask_1:.+]], %[[memory_w_mask_0:.+]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // FLATTEN:    %[[v17:.+]] = firrtl.cat %[[memory_w_mask_2:.+]], %[[v16]] : (!firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<3>
  // FLATTEN:    %[[v18:.+]] = firrtl.cat %[[memory_w_mask_3:.+]], %[[v17]] : (!firrtl.uint<1>, !firrtl.uint<3>) -> !firrtl.uint<4>
  // FLATTEN:    firrtl.connect %[[v15]], %[[v18]]
  // FLATTEN:    firrtl.connect %rData_0, %[[memory_r_data_0]]
  // FLATTEN:    firrtl.connect %rData_1, %[[memory_r_data_1]]
  // FLATTEN:    firrtl.connect %rData_2, %[[memory_r_data_2]]
  // FLATTEN:    firrtl.connect %rData_3, %[[memory_r_data_3]]
  // FLATTEN:    firrtl.connect %[[memory_w_mask_0]], %wMask_0
  // FLATTEN:    firrtl.connect %[[memory_w_mask_1]], %wMask_1
  // FLATTEN:    firrtl.connect %[[memory_w_mask_2]], %wMask_2
  // FLATTEN:    firrtl.connect %[[memory_w_mask_3]], %wMask_3
  // FLATTEN:    firrtl.connect %[[memory_w_data_0]], %wData_0
  // FLATTEN:    firrtl.connect %[[memory_w_data_1]], %wData_1
  // FLATTEN:    firrtl.connect %[[memory_w_data_2]], %wData_2
  // FLATTEN:    firrtl.connect %[[memory_w_data_3]], %wData_3


// Test that empty bundles can be lowered.
  // FLATTEN-LABEL: firrtl.module @SiFive_Queue_8()
  firrtl.module @SiFive_Queue_8()  {
    %SiFive_ram_MPORT, %SiFive_ram_io_deq_bits_MPORT = firrtl.mem Undefined  {depth = 2 : i64, name = "SiFive_ram", portNames = ["MPORT", "io_deq_bits_MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<id: uint<4>, addr: uint<31>, len: uint<8>, size: uint<3>, burst: uint<2>, lock: uint<1>, cache: uint<4>, prot: uint<3>, qos: uint<4>, user: bundle<>, echo: bundle<>>, mask: bundle<id: uint<1>, addr: uint<1>, len: uint<1>, size: uint<1>, burst: uint<1>, lock: uint<1>, cache: uint<1>, prot: uint<1>, qos: uint<1>, user: bundle<>, echo: bundle<>>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: bundle<id: uint<4>, addr: uint<31>, len: uint<8>, size: uint<3>, burst: uint<2>, lock: uint<1>, cache: uint<4>, prot: uint<3>, qos: uint<4>, user: bundle<>, echo: bundle<>>>
    // FLATTEN: %SiFive_ram_MPORT, %SiFive_ram_io_deq_bits_MPORT = firrtl.mem Undefined  {depth = 2 : i64, name = "SiFive_ram", portNames = ["MPORT", "io_deq_bits_MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<60>, mask: uint<60>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<60>>
   // FLATTEN:   %[[v3:.+]] = firrtl.subfield %SiFive_ram_MPORT(3) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<60>, mask: uint<60>>) -> !firrtl.uint<60>
   // FLATTEN:   %[[v4:.+]] = firrtl.cat %[[SiFive_ram_MPORT_data_addr:.+]], %[[SiFive_ram_MPORT_data_id:.+]] : (!firrtl.uint<31>, !firrtl.uint<4>) -> !firrtl.uint<35>
   // FLATTEN:   %[[v5:.+]] = firrtl.cat %[[SiFive_ram_MPORT_data_len:.+]], %[[v4]] : (!firrtl.uint<8>, !firrtl.uint<35>) -> !firrtl.uint<43>
   // FLATTEN:   %[[v6:.+]] = firrtl.cat %[[SiFive_ram_MPORT_data_size:.+]], %[[v5]] : (!firrtl.uint<3>, !firrtl.uint<43>) -> !firrtl.uint<46>
   // FLATTEN:   %[[v7:.+]] = firrtl.cat %[[SiFive_ram_MPORT_data_burst:.+]], %[[v6]] : (!firrtl.uint<2>, !firrtl.uint<46>) -> !firrtl.uint<48>
   // FLATTEN:   %[[v8:.+]] = firrtl.cat %[[SiFive_ram_MPORT_data_lock:.+]], %[[v7]] : (!firrtl.uint<1>, !firrtl.uint<48>) -> !firrtl.uint<49>
   // FLATTEN:   %[[v9:.+]] = firrtl.cat %[[SiFive_ram_MPORT_data_cache:.+]], %[[v8]] : (!firrtl.uint<4>, !firrtl.uint<49>) -> !firrtl.uint<53>
   // FLATTEN:   %[[v10:.+]] = firrtl.cat %[[SiFive_ram_MPORT_data_prot:.+]], %[[v9]] : (!firrtl.uint<3>, !firrtl.uint<53>) -> !firrtl.uint<56>
   // FLATTEN:   %[[v11:.+]] = firrtl.cat %[[SiFive_ram_MPORT_data_qos:.+]], %[[v10]] : (!firrtl.uint<4>, !firrtl.uint<56>) -> !firrtl.uint<60>
   // FLATTEN:   firrtl.connect %[[v3]], %[[v11]] : !firrtl.uint<60>, !firrtl.uint<60>
  }

// Issue #2315: Narrow index constants overflow when subaccessing long vectors.
// https://github.com/llvm/circt/issues/2315
// CHECK-LABEL: firrtl.module @Issue2315
firrtl.module @Issue2315(in %x: !firrtl.vector<uint<10>, 5>, in %source: !firrtl.uint<2>, out %z: !firrtl.uint<10>) {
  %0 = firrtl.subaccess %x[%source] : !firrtl.vector<uint<10>, 5>, !firrtl.uint<2>
  firrtl.connect %z, %0 : !firrtl.uint<10>, !firrtl.uint<10>
  // The width of multibit mux index will be converted at LowerToHW,
  // so it is ok that the type of `%source` is uint<2> here.
  // CHECK:      %0 = firrtl.multibit_mux %source, %x_4, %x_3, %x_2, %x_1, %x_0 : !firrtl.uint<2>, !firrtl.uint<10>
  // CHECK-NEXT: firrtl.connect %z, %0 : !firrtl.uint<10>, !firrtl.uint<10>
}

  // Check if the NLA is updated with the new lowered symbol on a field element.
  // CHECK-LABEL: firrtl.nla @nla
  firrtl.nla @nla [#hw.innerNameRef<@fallBackName::@test>, #hw.innerNameRef<@Aardvark::@test>, #hw.innerNameRef<@Zebra::@b>]
  // CHECK-SAME: [#hw.innerNameRef<@fallBackName::@test>, #hw.innerNameRef<@Aardvark::@test>, #hw.innerNameRef<@Zebra::@b_ready>]
  firrtl.nla @nla_1 [#hw.innerNameRef<@fallBackName::@test>,#hw.innerNameRef<@Aardvark::@test_1>, @Zebra]
  firrtl.module @fallBackName() {
    firrtl.instance test  sym @test {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}, {circt.nonlocal = @nla_1, class = "circt.nonlocal"} ]}@Aardvark()
    firrtl.instance test2 @Zebra()
  }

  firrtl.module @Aardvark() {
    firrtl.instance test sym @test {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}]}@Zebra()
    firrtl.instance test1 sym @test_1 {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}]}@Zebra()
  }

  // CHECK-LABEL: firrtl.module @Zebra()
  firrtl.module @Zebra(){
    %bundle = firrtl.wire sym @b {annotations = [#firrtl<"subAnno<fieldID = 2, {circt.nonlocal = @nla, class =\"test\" }>">]}: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
   // CHECK:   %bundle_valid = firrtl.wire sym @b_valid : !firrtl.uint<1>
   // CHECK-NEXT:   %bundle_ready = firrtl.wire sym @b_ready {annotations = [{circt.nonlocal = @nla, class = "test"}]} : !firrtl.uint<1>
   // CHECK-NEXT:   %bundle_data = firrtl.wire sym @b_data : !firrtl.uint<64>
  }

// Test the update of NLA when a new symbol is added after lowering of bundle fields.
  firrtl.nla @lowernla_2 [#hw.innerNameRef<@testNLAbundle::@testBundle_Bar>, #hw.innerNameRef<@testBundle_Bar::@d>]
  firrtl.nla @lowernla_1 [#hw.innerNameRef<@testNLAbundle::@testBundle_Bar>, #hw.innerNameRef<@testBundle_Bar::@b>]
  // CHECK: firrtl.nla @lowernla_2 [#hw.innerNameRef<@testNLAbundle::@testBundle_Bar>, #hw.innerNameRef<@testBundle_Bar::@d_qux>]
  // CHECK: firrtl.nla @lowernla_1 [#hw.innerNameRef<@testNLAbundle::@testBundle_Bar>, #hw.innerNameRef<@testBundle_Bar::@b_qux>]
  firrtl.module @testBundle_Bar(in %a: !firrtl.uint<1>, out %b: !firrtl.bundle<baz: uint<1>, qux: uint<1>> sym @b [#firrtl.subAnno<fieldID = 2, {circt.nonlocal = @lowernla_1, three}>], out %c: !firrtl.uint<1>) {
  // CHECK-LABEL: firrtl.module @testBundle_Bar
  // CHECK-SAME: out %b_qux: !firrtl.uint<1> sym @b_qux [{circt.nonlocal = @lowernla_1, three}]
    %d = firrtl.wire sym @d  {annotations = [#firrtl.subAnno<fieldID = 2, {circt.nonlocal = @lowernla_2, five}>]} : !firrtl.bundle<baz: uint<1>, qux: uint<1>>
  // CHECK:   %d_qux = firrtl.wire sym @d_qux  {annotations = [{circt.nonlocal = @lowernla_2, five}]} : !firrtl.uint<1>
  }
  firrtl.module @testNLAbundle() {
    %testBundle_Bar_a, %testBundle_Bar_b, %testBundle_Bar_c = firrtl.instance testBundle_Bar sym @testBundle_Bar  {annotations = [{circt.nonlocal = @lowernla_1, class = "circt.nonlocal"}, {circt.nonlocal = @lowernla_2, class = "circt.nonlocal"}]} @testBundle_Bar(in a: !firrtl.uint<1> [{one}], out b: !firrtl.bundle<baz: uint<1>, qux: uint<1>> [#firrtl.subAnno<fieldID = 1, {two}>], out c: !firrtl.uint<1> [{four}])
  }
} // CIRCUIT
