// RUN: circt-opt -pass-pipeline='firrtl.circuit(lower-firrtl-types)' %s | FileCheck %s

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

  // CHECK-LABEL: firrtl.module @Uniquification
  // CHECK-SAME: %[[FLATTENED_ARG:a_b]]: [[FLATTENED_TYPE:!firrtl.uint<1>]],
  // CHECK-NOT: %[[FLATTENED_ARG]]
  // CHECK-SAME: %[[RENAMED_ARG:a_b.+]]: [[RENAMED_TYPE:!firrtl.uint<1>]] {firrtl.name = "[[FLATTENED_ARG]]"}
  firrtl.module @Uniquification(%a: !firrtl.bundle<b: uint<1>>, %a_b: !firrtl.uint<1>) {
  }

  // CHECK-LABEL: firrtl.module @TopLevel
  // CHECK-SAME: %[[SOURCE_VALID_NAME:source_valid]]: [[SOURCE_VALID_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: %[[SOURCE_READY_NAME:source_ready]]: [[SOURCE_READY_TYPE:!firrtl.flip<uint<1>>]]
  // CHECK-SAME: %[[SOURCE_DATA_NAME:source_data]]: [[SOURCE_DATA_TYPE:!firrtl.uint<64>]]
  // CHECK-SAME: %[[SINK_VALID_NAME:sink_valid]]: [[SINK_VALID_TYPE:!firrtl.flip<uint<1>>]]
  // CHECK-SAME: %[[SINK_READY_NAME:sink_ready]]: [[SINK_READY_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: %[[SINK_DATA_NAME:sink_data]]: [[SINK_DATA_TYPE:!firrtl.flip<uint<64>>]]
  firrtl.module @TopLevel(%source: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>,
                          %sink: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) {

    // CHECK-NEXT: %[[INSTANCE:.+]] = firrtl.instance @Simple
    // CHECK-SAME: !firrtl.bundle
    // CHECK-SAME: [[SUB_SOURCE_VALID_NAME:source_valid]]: [[SUB_SOURCE_VALID_TYPE:flip<uint<1>>]]
    // CHECK-SAME: [[SUB_SOURCE_READY_NAME:source_ready]]: [[SUB_SOURCE_READY_TYPE:uint<1>]]
    // CHECK-SAME: [[SUB_SOURCE_DATA_NAME:source_data]]: [[SUB_SOURCE_DATA_TYPE:flip<uint<64>>]]
    // CHECK-SAME: [[SUB_SINK_VALID_NAME:sink_valid]]: [[SUB_SINK_VALID_TYPE:uint<1>]]
    // CHECK-SAME: [[SUB_SINK_READY_NAME:sink_ready]]: [[SUB_SINK_READY_TYPE:flip<uint<1>>]]
    // CHECK-SAME: [[SUB_SINK_DATA_NAME:sink_data]]: [[SUB_SINK_DATA_TYPE:uint<64>]]
    %0 = firrtl.instance @Simple {name = ""} : !firrtl.bundle<source: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, sink: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>

    // CHECK-NEXT: %[[FIELD1:.+]] = firrtl.subfield %[[INSTANCE]]("[[SUB_SOURCE_VALID_NAME]]") {{.*}} [[SUB_SOURCE_VALID_TYPE]]
    // CHECK-NEXT: %[[FIELD2:.+]] = firrtl.subfield %[[INSTANCE]]("[[SUB_SOURCE_READY_NAME]]") {{.*}} [[SUB_SOURCE_READY_TYPE]]
    // CHECK-NEXT: %[[FIELD3:.+]] = firrtl.subfield %[[INSTANCE]]("[[SUB_SOURCE_DATA_NAME]]") {{.*}} [[SUB_SOURCE_DATA_TYPE]]
    // CHECK-NEXT: firrtl.connect %[[FIELD1]], %[[SOURCE_VALID_NAME]] : !firrtl.[[SUB_SOURCE_VALID_TYPE]], [[SOURCE_VALID_TYPE]]
    // CHECK-NEXT: firrtl.connect %[[FIELD2]], %[[SOURCE_READY_NAME]] : !firrtl.[[SUB_SOURCE_READY_TYPE]], [[SOURCE_READY_TYPE]]
    // CHECK-NEXT: firrtl.connect %[[FIELD3]], %[[SOURCE_DATA_NAME]] : !firrtl.[[SUB_SOURCE_DATA_TYPE]], [[SOURCE_DATA_TYPE]]
    %1 = firrtl.subfield %0("source") : (!firrtl.bundle<source: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, sink: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>) -> !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>
    firrtl.connect %1, %source : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>

    // CHECK-NEXT: %[[FIELD4:.+]] = firrtl.subfield %[[INSTANCE]]("[[SUB_SINK_VALID_NAME]]") {{.*}} [[SUB_SINK_VALID_TYPE]]
    // CHECK-NEXT: %[[FIELD5:.+]] = firrtl.subfield %[[INSTANCE]]("[[SUB_SINK_READY_NAME]]") {{.*}} [[SUB_SINK_READY_TYPE]]
    // CHECK-NEXT: %[[FIELD6:.+]] = firrtl.subfield %[[INSTANCE]]("[[SUB_SINK_DATA_NAME]]") {{.*}} [[SUB_SINK_DATA_TYPE]]
    // CHECK-NEXT: firrtl.connect %[[SINK_VALID_NAME]], %[[FIELD4]] : [[SINK_VALID_TYPE]], !firrtl.[[SUB_SINK_VALID_TYPE]]
    // CHECK-NEXT: firrtl.connect %[[SINK_READY_NAME]], %[[FIELD5]] : [[SINK_READY_TYPE]], !firrtl.[[SUB_SINK_READY_TYPE]]
    // CHECK-NEXT: firrtl.connect %[[SINK_DATA_NAME]], %[[FIELD6]] : [[SINK_DATA_TYPE]], !firrtl.[[SUB_SINK_DATA_TYPE]]
    %2 = firrtl.subfield %0("sink") : (!firrtl.bundle<source: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, sink: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>) -> !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
    firrtl.connect %sink, %2 : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
  }
}
