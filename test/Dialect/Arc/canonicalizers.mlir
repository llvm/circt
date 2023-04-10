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

// CHECK-LABEL: hw.module @clockDomainDCE
hw.module @clockDomainDCE(%clk: i1) {
  // CHECK-NOT: arc.clock_domain
  arc.clock_domain () clock %clk : () -> () {
  ^bb0:
    arc.output
  }
}

// CHECK-LABEL: hw.module @memoryOps
hw.module @memoryOps(%clk: i1, %mem: !arc.memory<4 x i32>, %addr: i32, %data: i32) -> (out0: i32, out1: i32) {
  %true = hw.constant true
  // CHECK: [[RD:%.+]] = arc.memory_read_port %mem[%addr] clock %clk : <4 x i32>, i32
  %0 = arc.memory_read_port %mem[%addr] if %true clock %clk : <4 x i32>, i32
  // CHECK-NEXT: arc.memory_write_port %mem[%addr], %data clock %clk : <4 x i32>, i32
  arc.memory_write_port %mem[%addr], %data if %true clock %clk : <4 x i32>, i32
  // CHECK-NEXT: arc.memory_write %mem[%addr], %data : <4 x i32>, i32
  arc.memory_write %mem[%addr], %data if %true : <4 x i32>, i32

  %false = hw.constant false
  %1 = arc.memory_read_port %mem[%addr] if %false clock %clk : <4 x i32>, i32
  arc.memory_write_port %mem[%addr], %data if %false clock %clk : <4 x i32>, i32
  arc.memory_write %mem[%addr], %data if %false : <4 x i32>, i32

  // CHECK-NEXT: hw.output [[RD]], %c0_i32
  hw.output %0, %1 : i32, i32
}
