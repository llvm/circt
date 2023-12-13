// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-passive-wires)))' --allow-unregistered-dialect %s | FileCheck --check-prefixes=CHECK %s

firrtl.circuit "TopLevel" {

  // CHECK-LABEL: @TopLevel
  firrtl.module @TopLevel(in %source: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
                             out %sink: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
    %w = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    firrtl.connect %w, %source : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    firrtl.connect %sink, %w : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>

// CHECK: %w = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>> 
// CHECK: %0 = firrtl.subfield %w[valid] : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
// CHECK: %1 = firrtl.subfield %source[valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK: firrtl.strictconnect %0, %1 : !firrtl.uint<1>
// CHECK: %2 = firrtl.subfield %w[ready] : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
// CHECK: %3 = firrtl.subfield %source[ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK: firrtl.strictconnect %3, %2 : !firrtl.uint<1>
// CHECK: %4 = firrtl.subfield %w[data] : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
// CHECK: %5 = firrtl.subfield %source[data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK: firrtl.strictconnect %4, %5 : !firrtl.uint<64>
// CHECK: %6 = firrtl.subfield %sink[valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK: %7 = firrtl.subfield %w[valid] : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
// CHECK: firrtl.strictconnect %6, %7 : !firrtl.uint<1>
// CHECK: %8 = firrtl.subfield %sink[ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK: %9 = firrtl.subfield %w[ready] : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
// CHECK: firrtl.strictconnect %9, %8 : !firrtl.uint<1>
// CHECK: %10 = firrtl.subfield %sink[data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK: %11 = firrtl.subfield %w[data] : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
// CHECK: firrtl.strictconnect %10, %11 : !firrtl.uint<64>
  }


  // CHECK-LABEL: firrtl.module private @SendRefTypeVectors1
  firrtl.module private @SendRefTypeVectors1(out %b: !firrtl.probe<bundle<b : vector<uint<1>, 2>>>) {
    %a = firrtl.wire : !firrtl.bundle<b flip : vector<uint<1>, 2>>
    %0 = firrtl.ref.send %a : !firrtl.bundle<b flip : vector<uint<1>, 2>>
    firrtl.ref.define %b, %0 : !firrtl.probe<bundle<b : vector<uint<1>, 2>>>

    // CHECK:  %a = firrtl.wire : !firrtl.bundle<b: vector<uint<1>, 2>>
    // CHECK: %0 = firrtl.ref.send %a : !firrtl.bundle<b: vector<uint<1>, 2>>
    // CHECK: firrtl.ref.define %b, %0 : !firrtl.probe<bundle<b: vector<uint<1>, 2>>>
  }

}