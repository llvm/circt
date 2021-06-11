// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-lower-bundle-vectors)' -split-input-file %s | FileCheck %s

firrtl.circuit "Simple" {

  // CHECK-LABEL: firrtl.module @Simple
  // CHECK-SAME: in %[[SOURCE_VALID_NAME:source_valid]]: [[SOURCE_VALID_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: out %[[SOURCE_READY_NAME:source_ready]]: [[SOURCE_READY_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: in %[[SOURCE_DATA_NAME:source_data]]: [[SOURCE_DATA_TYPE:!firrtl.uint<64>]]
  // CHECK-SAME: out %[[SINK_VALID_NAME:sink_valid]]: [[SINK_VALID_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: in %[[SINK_READY_NAME:sink_ready]]: [[SINK_READY_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: out %[[SINK_DATA_NAME:sink_data]]: [[SINK_DATA_TYPE:!firrtl.uint<64>]]
  firrtl.module @Simple(in %source: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>,
                        out %sink: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) {

    // CHECK-NEXT: firrtl.when %[[SOURCE_VALID_NAME]]
    // CHECK-NEXT:   firrtl.connect %[[SINK_DATA_NAME]], %[[SOURCE_DATA_NAME]] : [[SINK_DATA_TYPE]], [[SOURCE_DATA_TYPE]]
    // CHECK-NEXT:   firrtl.connect %[[SINK_VALID_NAME]], %[[SOURCE_VALID_NAME]] : [[SINK_VALID_TYPE]], [[SOURCE_VALID_TYPE]]
    // CHECK-NEXT:   firrtl.connect %[[SOURCE_READY_NAME]], %[[SINK_READY_NAME]] : [[SOURCE_READY_TYPE]], [[SINK_READY_TYPE]]

    %0 = firrtl.subfield %source("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %source("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
    %2 = firrtl.subfield %source("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
    %3 = firrtl.subfield %sink("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
    %4 = firrtl.subfield %sink("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
    %5 = firrtl.subfield %sink("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
    firrtl.when %0 {
      firrtl.connect %5, %2 : !firrtl.uint<64>, !firrtl.uint<64>
      firrtl.connect %3, %0 : !firrtl.uint<1>, !firrtl.uint<1>
      firrtl.connect %1, %4 : !firrtl.uint<1>, !firrtl.uint<1>
    }
  }
}

// -----

firrtl.circuit "Top" {

  // CHECK-LABEL: firrtl.module @Top
  firrtl.module @Top(in %in : !firrtl.bundle<a: uint<1>, b: uint<1>>,
                     out %out : !firrtl.bundle<a: uint<1>, b: uint<1>>) {
    // CHECK: firrtl.connect %out_a, %in_a : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.connect %out_b, %in_b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %in : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
  }

}
