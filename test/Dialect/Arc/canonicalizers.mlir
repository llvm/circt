// RUN: circt-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: hw.module @stateOpCanonicalizer
hw.module @stateOpCanonicalizer(%clk: i1, %in: i32) {
  arc.state @Foo(%in) clock %clk lat 1 : (i32) -> ()
  %0 = arc.state @Bar(%in) lat 0 : (i32) -> (i32)
  // CHECK-NEXT: {{%.+}} = arc.state @Bar(%in) clock %clk lat 1 {name = "stateName"} : (i32) -> i32
  %1 = arc.state @Bar(%in) clock %clk lat 1 {name = "stateName"} : (i32) -> i32
  // CHECK-NEXT: {{%.+}} = arc.state @Bar(%in) clock %clk lat 1 {names = ["stateName"]} : (i32) -> i32
  %2 = arc.state @Bar(%in) clock %clk lat 1 {names = ["stateName"]} : (i32) -> i32
  // CHECK-NEXT: hw.output
}
arc.define @Foo(%arg0: i32) {
  arc.output
}
arc.define @Bar(%arg0: i32) -> i32{
  %c0_i32 = hw.constant 0 : i32
  arc.output %c0_i32 : i32
}
