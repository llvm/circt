// RUN: circt-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: hw.module @stateOpCanonicalizer
hw.module @stateOpCanonicalizer(%clk: i1, %in: i32, %en: i1, %rst: i1) -> (out0: i32, out1: i32, out2: i32, out3: i32, out4: i32, out5: i32, out6: i32, out7: i32, out8: i32, out9: i32, out10: i32, out11: i32, out12: i32, out13: i32, out14: i32, out15: i32) {
  // CHECK-NEXT: %c0_i32 = hw.constant 0 : i32
  // CHECK-NEXT: %true = hw.constant true
  // CHECK-NEXT: %false = hw.constant false
  %true = hw.constant true
  %false = hw.constant false

  arc.state @Foo(%in) clock %clk lat 1 : (i32) -> ()
  %0 = arc.state @Bar(%in) lat 0 : (i32) -> (i32)
  // CHECK-NEXT: {{%.+}} = arc.state @Bar(%in) clock %clk lat 1 {name = "stateName"} : (i32) -> i32
  %1 = arc.state @Bar(%in) clock %clk lat 1 {name = "stateName"} : (i32) -> i32
  // CHECK-NEXT: {{%.+}} = arc.state @Bar(%in) clock %clk lat 1 {names = ["stateName"]} : (i32) -> i32
  %2 = arc.state @Bar(%in) clock %clk lat 1 {names = ["stateName"]} : (i32) -> i32

  %3 = arc.state @Passthrough(%in) clock %clk enable %false reset %false lat 1 : (i32) -> i32
  // CHECK-NEXT: [[V4:%.+]] = arc.state @Passthrough(%in) clock %clk lat 1 : (i32) -> i32
  %4 = arc.state @Passthrough(%in) clock %clk enable %true reset %false lat 1 : (i32) -> i32
  %5 = arc.state @Passthrough(%in) clock %clk enable %false reset %true lat 1 : (i32) -> i32
  %6 = arc.state @Passthrough(%in) clock %clk enable %true reset %true lat 1 : (i32) -> i32

  %7 = arc.state @Passthrough(%in) clock %clk enable %false reset %rst lat 1 : (i32) -> i32
  // CHECK-NEXT: [[V8:%.+]] = arc.state @Passthrough(%in) clock %clk reset %rst lat 1 : (i32) -> i32
  %8 = arc.state @Passthrough(%in) clock %clk enable %true reset %rst lat 1 : (i32) -> i32
  // CHECK-NEXT: [[V9:%.+]] = arc.state @Passthrough(%in) clock %clk enable %en lat 1 : (i32) -> i32
  %9 = arc.state @Passthrough(%in) clock %clk enable %en reset %false lat 1 : (i32) -> i32
  %10 = arc.state @Passthrough(%in) clock %clk enable %en reset %true lat 1 : (i32) -> i32

  %11:2 = arc.state @Passthrough2(%in, %in) clock %clk enable %false lat 1 : (i32, i32) -> (i32, i32)
  // CHECK-NEXT: [[V12:%.+]]:2 = arc.state @Passthrough2(%in, %in) clock %clk lat 1 : (i32, i32) -> (i32, i32)
  %12:2 = arc.state @Passthrough2(%in, %in) clock %clk enable %true lat 1 : (i32, i32) -> (i32, i32)
  // CHECK-NEXT: [[V13:%.+]]:2 = arc.state @Passthrough2(%in, %in) clock %clk lat 1 : (i32, i32) -> (i32, i32)
  %13:2 = arc.state @Passthrough2(%in, %in) clock %clk reset %false lat 1 : (i32, i32) -> (i32, i32)
  %14:2 = arc.state @Passthrough2(%in, %in) clock %clk reset %true lat 1 : (i32, i32) -> (i32, i32)

  // CHECK-NEXT: %{{.+}} = arc.state @Passthrough(%in) clock %clk enable %false reset %true lat 1 {name = "stateName"} : (i32) -> i32
  %15 = arc.state @Passthrough(%in) clock %clk enable %false reset %true lat 1 {name = "stateName"} : (i32) -> i32
  // CHECK-NEXT: %{{.+}} = arc.state @Passthrough(%in) clock %clk enable %false reset %true lat 1 {names = ["stateName"]} : (i32) -> i32
  %16 = arc.state @Passthrough(%in) clock %clk enable %false reset %true lat 1 {names = ["stateName"]} : (i32) -> i32

  // CHECK-NEXT: hw.output %c0_i32, [[V4]], %c0_i32, %c0_i32, %c0_i32, [[V8]], [[V9]], %c0_i32, %c0_i32, %c0_i32, [[V12]]#0, [[V12]]#1, [[V13]]#0, [[V13]]#1, %c0_i32, %c0_i32 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
  hw.output %3, %4, %5, %6, %7, %8, %9, %10, %11#0, %11#1, %12#0, %12#1, %13#0, %13#1, %14#0, %14#1 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
}
arc.define @Foo(%arg0: i32) {
  arc.output
}
arc.define @Bar(%arg0: i32) -> i32{
  %c0_i32 = hw.constant 0 : i32
  arc.output %c0_i32 : i32
}
arc.define @Passthrough(%arg0: i32) -> i32 {
  arc.output %arg0 : i32
}
arc.define @Passthrough2(%arg0: i32, %arg1: i32) -> (i32, i32) {
  arc.output %arg0, %arg1 : i32, i32
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
