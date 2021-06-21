// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @std_addi_2ins_1outs_ui64(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK:   %0 = firrtl.subfield %arg0("valid") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %1 = firrtl.subfield %arg0("ready") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %2 = firrtl.subfield %arg0("data") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %3 = firrtl.subfield %arg1("valid") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %4 = firrtl.subfield %arg1("ready") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %5 = firrtl.subfield %arg1("data") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %6 = firrtl.subfield %arg2("valid") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %7 = firrtl.subfield %arg2("ready") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %8 = firrtl.subfield %arg2("data") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %9 = firrtl.add %2, %5 : (!firrtl.uint<64>, !firrtl.uint<64>) -> !firrtl.uint<65>
// CHECK:   %10 = firrtl.bits %9 63 to 0 : (!firrtl.uint<65>) -> !firrtl.uint<64>
// CHECK:   firrtl.connect %8, %10 : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK:   %11 = firrtl.and %0, %3 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %6, %11 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   %12 = firrtl.and %7, %11 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %1, %12 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   firrtl.connect %4, %12 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK: }

// CHECK-LABEL: firrtl.module @simple_addi(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
// CHECK-SAME:  in %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
// CHECK-SAME:  in %arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>,
// CHECK-SAME:  out %arg3: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
// CHECK-SAME:  out %arg4: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>,
// CHECK-SAME:  in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
handshake.func @simple_addi(%arg0: index, %arg1: index, %arg2: none, ...) -> (index, none) {

  // CHECK:%inst_arg0, %inst_arg1, %inst_arg2 = firrtl.instance @std_addi_2ins_1outs_ui64  {name = ""} : !firrtl.flip<bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>>, !firrtl.flip<bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
  // CHECK: firrtl.connect %inst_arg0, %arg0 : !firrtl.flip<bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
  // CHECK: firrtl.connect %inst_arg1, %arg1 : !firrtl.flip<bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
  %0 = addi %arg0, %arg1 : index

  // CHECK: firrtl.connect %arg3, %inst_arg2 : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
  // CHECK: firrtl.connect %arg4, %arg2 : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
  handshake.return %0, %arg2 : index, none
}
