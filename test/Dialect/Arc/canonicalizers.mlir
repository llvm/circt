// RUN: circt-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: hw.module @stateOpCanonicalizer
hw.module @stateOpCanonicalizer(in %clk: !seq.clock, in %in: i32, in %en: i1, in %rst: i1, out out0: i32, out out1: i32, out out2: i32, out out3: i32, out out4: i32, out out5: i32, out out6: i32, out out7: i32, out out8: i32, out out9: i32, out out10: i32, out out11: i32, out out12: i32, out out13: i32, out out14: i32, out out15: i32, out out16: i32, out out17: i32, out out18: i32, out out19: i32, out out18: i32, out out19: i32) {
  // CHECK-NEXT: %c0_i32 = hw.constant 0 : i32
  // CHECK-NEXT: %true = hw.constant true
  // CHECK-NEXT: %false = hw.constant false
  // CHECK-DAG: %c1234_i32 = hw.constant 1234 : i32
  // CHECK-DAG: %c4321_i32 = hw.constant 4321 : i32

  %true = hw.constant true
  %false = hw.constant false
  %c1234_i32 = hw.constant 1234 : i32
  %c4321_i32 = hw.constant 4321 : i32

  arc.state @Foo(%in) clock %clk latency 1 : (i32) -> ()
  arc.state @Foo(%in) clock %clk enable %false latency 1 : (i32) -> ()

  // CHECK-NEXT: {{%.+}} = arc.state @Bar(%in) clock %clk latency 1 {name = "stateName"} : (i32) -> i32
  %1 = arc.state @Bar(%in) clock %clk latency 1 {name = "stateName"} : (i32) -> i32
  // CHECK-NEXT: {{%.+}} = arc.state @Bar(%in) clock %clk latency 1 {names = ["stateName"]} : (i32) -> i32
  %2 = arc.state @Bar(%in) clock %clk latency 1 {names = ["stateName"]} : (i32) -> i32

  %3 = arc.state @Passthrough(%in) clock %clk enable %false reset %false latency 1 : (i32) -> i32
  // CHECK-NEXT: [[V4:%.+]] = arc.state @Passthrough(%in) clock %clk latency 1 : (i32) -> i32
  %4 = arc.state @Passthrough(%in) clock %clk enable %true reset %false latency 1 : (i32) -> i32
  %5 = arc.state @Passthrough(%in) clock %clk enable %false reset %true latency 1 : (i32) -> i32
  %6 = arc.state @Passthrough(%in) clock %clk enable %true reset %true latency 1 : (i32) -> i32

  %7 = arc.state @Passthrough(%in) clock %clk enable %false reset %rst latency 1 : (i32) -> i32
  // CHECK-NEXT: [[V8:%.+]] = arc.state @Passthrough(%in) clock %clk reset %rst latency 1 : (i32) -> i32
  %8 = arc.state @Passthrough(%in) clock %clk enable %true reset %rst latency 1 : (i32) -> i32
  // CHECK-NEXT: [[V9:%.+]] = arc.state @Passthrough(%in) clock %clk enable %en latency 1 : (i32) -> i32
  %9 = arc.state @Passthrough(%in) clock %clk enable %en reset %false latency 1 : (i32) -> i32
  %10 = arc.state @Passthrough(%in) clock %clk enable %en reset %true latency 1 : (i32) -> i32

  %11:2 = arc.state @Passthrough2(%in, %in) clock %clk enable %false latency 1 : (i32, i32) -> (i32, i32)
  // CHECK-NEXT: [[V12:%.+]]:2 = arc.state @Passthrough2(%in, %in) clock %clk latency 1 : (i32, i32) -> (i32, i32)
  %12:2 = arc.state @Passthrough2(%in, %in) clock %clk enable %true latency 1 : (i32, i32) -> (i32, i32)
  // CHECK-NEXT: [[V13:%.+]]:2 = arc.state @Passthrough2(%in, %in) clock %clk latency 1 : (i32, i32) -> (i32, i32)
  %13:2 = arc.state @Passthrough2(%in, %in) clock %clk reset %false latency 1 : (i32, i32) -> (i32, i32)
  %14:2 = arc.state @Passthrough2(%in, %in) clock %clk reset %true latency 1 : (i32, i32) -> (i32, i32)

  // CHECK-NEXT: %{{.+}} = arc.state @Passthrough(%in) clock %clk enable %false reset %true latency 1 {name = "stateName"} : (i32) -> i32
  %15 = arc.state @Passthrough(%in) clock %clk enable %false reset %true latency 1 {name = "stateName"} : (i32) -> i32
  // CHECK-NEXT: %{{.+}} = arc.state @Passthrough(%in) clock %clk enable %false reset %true latency 1 {names = ["stateName"]} : (i32) -> i32
  %16 = arc.state @Passthrough(%in) clock %clk enable %false reset %true latency 1 {names = ["stateName"]} : (i32) -> i32

  %17:2 = arc.state @Passthrough2(%in, %in) clock %clk enable %false reset %true initial (%c1234_i32, %c4321_i32 : i32, i32) latency 1 : (i32, i32) -> (i32, i32)
  // CHECK-NEXT: [[V14:%.+]]:2 = arc.state @Passthrough2(%in, %in) clock %clk enable %false reset %true initial (%c1234_i32, %in : i32, i32) latency 1 : (i32, i32) -> (i32, i32)
  %18:2 = arc.state @Passthrough2(%in, %in) clock %clk enable %false reset %true initial (%c1234_i32, %in : i32, i32) latency 1 : (i32, i32) -> (i32, i32)

  // CHECK-NEXT: [[V15:%.+]]:2 = arc.state @Passthrough2(%in, %in) clock %clk enable %en reset %true initial (%c4321_i32, %c1234_i32 : i32, i32) latency 1 : (i32, i32) -> (i32, i32)
  %19:2 = arc.state @Passthrough2(%in, %in) clock %clk enable %en reset %true initial (%c4321_i32, %c1234_i32 : i32, i32) latency 1 : (i32, i32) -> (i32, i32)

  // CHECK-NEXT: hw.output %c0_i32, [[V4]], %c0_i32, %c0_i32, %c0_i32, [[V8]], [[V9]], %c0_i32, %c0_i32, %c0_i32, [[V12]]#0, [[V12]]#1, [[V13]]#0, [[V13]]#1, %c0_i32, %c0_i32, %c1234_i32, %c4321_i32, [[V14]]#0, [[V14]]#1, [[V15]]#0, [[V15]]#1 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
  hw.output %3, %4, %5, %6, %7, %8, %9, %10, %11#0, %11#1, %12#0, %12#1, %13#0, %13#1, %14#0, %14#1, %17#0, %17#1, %18#0, %18#1, %19#0, %19#1 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
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

// CHECK-LABEL: arc.model @StorageGetCanonicalizers
arc.model @StorageGetCanonicalizers io !hw.modty<> {
// CHECK-NEXT: ^bb
^bb0(%arg0: !arc.storage):
  %0 = arc.storage.get %arg0[16] : !arc.storage -> !arc.storage
  %1 = arc.storage.get %0[0] : !arc.storage -> !arc.state<i64>
  %2 = arc.storage.get %0[8] : !arc.storage -> !arc.state<i64>
  %3 = arc.state_read %1 : <i64>
  arc.state_write %2 = %3 : <i64>
  // CHECK-NEXT: [[V0:%.+]] = arc.storage.get %arg0[16] : !arc.storage -> !arc.state<i64>
  // CHECK-NEXT: [[V1:%.+]] = arc.storage.get %arg0[24] : !arc.storage -> !arc.state<i64>
  // CHECK-NEXT: [[V2:%.+]] = arc.state_read [[V0]] : <i64>
  // CHECK-NEXT: arc.state_write [[V1]] = [[V2]] : <i64>
}
