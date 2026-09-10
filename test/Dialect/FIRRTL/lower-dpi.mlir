// RUN: circt-opt -firrtl-lower-dpi %s | FileCheck %s

// CHECK-LABEL: firrtl.circuit "DPI" {
firrtl.circuit "DPI" {
  // CHECK-NEXT: sim.func.dpi private @unclocked_result(in %in_0 : i8, in %in_1 : i8, out out_0 : i8) attributes {verilogName = "unclocked_result"}
  // CHECK-NEXT: sim.func.dpi private @clocked_void(in %in_0 : i8, in %in_1 : i8) attributes {verilogName = "clocked_void"}
  // CHECK-NEXT: sim.func.dpi private @clocked_result(in %foo : i8, in %bar : i8, out baz : i8) attributes {verilogName = "clocked_result"}
  // CHECK-LABEL: firrtl.module @DPI
  firrtl.module @DPI(in %clock: !firrtl.clock, in %enable: !firrtl.uint<1>, in %in_0: !firrtl.uint<8>, in %in_1: !firrtl.uint<8>, out %out_0: !firrtl.uint<8>, out %out_1: !firrtl.uint<8>) attributes {convention = #firrtl<convention scalarized>} {
    // CHECK-NEXT: %0 = builtin.unrealized_conversion_cast %clock : !firrtl.clock to !seq.clock
    // CHECK-NEXT: %1 = builtin.unrealized_conversion_cast %enable : !firrtl.uint<1> to i1
    // CHECK-NEXT: %2 = builtin.unrealized_conversion_cast %in_0 : !firrtl.uint<8> to i8
    // CHECK-NEXT: %3 = builtin.unrealized_conversion_cast %in_1 : !firrtl.uint<8> to i8
    // CHECK-NEXT: %4 = sim.func.dpi.call @clocked_result(%2, %3) clock %0 enable %1 : (i8, i8) -> i8
    // CHECK-NEXT: %5 = builtin.unrealized_conversion_cast %4 : i8 to !firrtl.uint<8>
    // CHECK-NEXT: %6 = builtin.unrealized_conversion_cast %clock : !firrtl.clock to !seq.clock
    // CHECK-NEXT: %7 = builtin.unrealized_conversion_cast %enable : !firrtl.uint<1> to i1
    // CHECK-NEXT: %8 = builtin.unrealized_conversion_cast %in_0 : !firrtl.uint<8> to i8
    // CHECK-NEXT: %9 = builtin.unrealized_conversion_cast %in_1 : !firrtl.uint<8> to i8
    // CHECK-NEXT: sim.func.dpi.call @clocked_void(%8, %9) clock %6 enable %7 : (i8, i8) -> ()
    // CHECK-NEXT: %10 = builtin.unrealized_conversion_cast %enable : !firrtl.uint<1> to i1
    // CHECK-NEXT: %11 = builtin.unrealized_conversion_cast %in_0 : !firrtl.uint<8> to i8
    // CHECK-NEXT: %12 = builtin.unrealized_conversion_cast %in_1 : !firrtl.uint<8> to i8
    // CHECK-NEXT: %13 = sim.func.dpi.call @unclocked_result(%11, %12) enable %10 : (i8, i8) -> i8
    // CHECK-NEXT: %14 = builtin.unrealized_conversion_cast %13 : i8 to !firrtl.uint<8>
    // CHECK-NEXT: firrtl.matchingconnect %out_0, %5 : !firrtl.uint<8>
    // CHECK-NEXT: firrtl.matchingconnect %out_1, %14 : !firrtl.uint<8>
    %0 = firrtl.int.dpi.call "clocked_result"(%in_0, %in_1) clock %clock enable %enable {inputNames = ["foo", "bar"], outputName = "baz"} : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
    firrtl.int.dpi.call "clocked_void"(%in_0, %in_1) clock %clock enable %enable : (!firrtl.uint<8>, !firrtl.uint<8>) -> ()
    %1 = firrtl.int.dpi.call "unclocked_result"(%in_0, %in_1) enable %enable {name = "result2"} : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
    firrtl.matchingconnect %out_0, %0 : !firrtl.uint<8>
    firrtl.matchingconnect %out_1, %1 : !firrtl.uint<8>
  }

  // CHECK-LABEL: firrtl.module @DPISignature
  firrtl.module @DPISignature(in %clock: !firrtl.clock, in %enable: !firrtl.uint<1>, in %in_0: !firrtl.uint<8>, in %in_1: !firrtl.uint<8>) attributes {convention = #firrtl<convention scalarized>} {
    // CHECK: call @clocked_void
    // CHECK: call @clocked_void
    firrtl.int.dpi.call "clocked_void"(%in_0, %in_1) clock %clock enable %enable : (!firrtl.uint<8>, !firrtl.uint<8>) -> ()
    firrtl.int.dpi.call "clocked_void"(%in_0, %in_1) clock %clock : (!firrtl.uint<8>, !firrtl.uint<8>) -> ()
  }
}
