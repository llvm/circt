// RUN: circt-opt %s -test-scheduling-problem -allow-unregistered-dialect
// RUN: circt-opt %s -test-asap-scheduler -allow-unregistered-dialect | FileCheck %s -check-prefix=ASAP
// RUN: circt-opt %s -test-simplex-scheduler=with=Problem -allow-unregistered-dialect | FileCheck %s -check-prefix=SIMPLEX

// ASAP-LABEL: unit_latencies
// SIMPLEX-LABEL: unit_latencies
func @unit_latencies(%a1 : i32, %a2 : i32, %a3 : i32, %a4 : i32) -> i32 {
  // ASAP-NEXT: asapStartTime = 0
  %0 = arith.addi %a1, %a2 { problemStartTime = 0 } : i32
  // ASAP-NEXT: asapStartTime = 1
  %1 = arith.addi %0, %a3 { problemStartTime = 1 } : i32
  // ASAP-NEXT: asapStartTime = 2
  %2:3 = "more.results"(%0, %1) { problemStartTime = 2 } : (i32, i32) -> (i32, i32, i32)
  // ASAP-NEXT: asapStartTime = 3
  %3 = arith.addi %a4, %2#1 { problemStartTime = 3 } : i32
  // ASAP-NEXT: asapStartTime = 3
  %4 = arith.addi %2#0, %2#2 { problemStartTime = 4 } : i32
  // ASAP-NEXT: asapStartTime = 4
  %5 = arith.addi %3, %3 { problemStartTime = 4 } : i32
  // ASAP-NEXT: asapStartTime = 5
  %6 = "more.operands"(%3, %4, %5) { problemStartTime = 6 } : (i32, i32, i32) -> i32
  // ASAP-NEXT: asapStartTime = 6
  // SIMPLEX: return
  // SIMPLEX-SAME: simplexStartTime = 6
  return { problemStartTime = 7 } %6 : i32
}

// ASAP-LABEL: arbitrary_latencies
// SIMPLEX-LABEL: arbitrary_latencies
func @arbitrary_latencies(%v : complex<f32>) -> f32 attributes {
  operatortypes = [
    { name = "extr", latency = 0 },
    { name = "add", latency = 3 },
    { name = "mult", latency = 6 },
    { name = "sqrt", latency = 10 }
  ] } {
  // ASAP-NEXT: asapStartTime = 0
  %0 = "complex.re"(%v) { opr = "extr", problemStartTime = 0 } : (complex<f32>) -> f32
  // ASAP-NEXT: asapStartTime = 0
  %1 = "complex.im"(%v) { opr = "extr", problemStartTime = 10 } : (complex<f32>) -> f32
  // ASAP-NEXT: asapStartTime = 0
  %2 = arith.mulf %0, %0 { opr = "mult", problemStartTime = 20 } : f32
  // ASAP-NEXT: asapStartTime = 0
  %3 = arith.mulf %1, %1 { opr = "mult", problemStartTime = 30 } : f32
  // ASAP-NEXT: asapStartTime = 6
  %4 = arith.addf %2, %3 { opr = "add", problemStartTime = 40 } : f32
  // ASAP-NEXT: asapStartTime = 9
  %5 = "math.sqrt"(%4) { opr = "sqrt", problemStartTime = 50 } : (f32) -> f32
  // ASAP-NEXT: asapStartTime = 19
  // SIMPLEX: return
  // SIMPLEX-SAME: simplexStartTime = 19
  return { problemStartTime = 60 } %5 : f32
}

// ASAP-LABEL: auxiliary_dependences
// SIMPLEX-LABEL: auxiliary_dependences
func @auxiliary_dependences() attributes { auxdeps = [
    [0,1], [0,2], [2,3], [3,4], [3,6], [4,5], [5,6]
  ] } {
  // ASAP-NEXT: asapStartTime = 0
  %0 = arith.constant { problemStartTime = 0 } 0 : i32
  // ASAP-NEXT: asapStartTime = 1
  %1 = arith.constant { problemStartTime = 1 } 1 : i32
  // ASAP-NEXT: asapStartTime = 1
  %2 = arith.constant { problemStartTime = 2 } 2 : i32
  // ASAP-NEXT: asapStartTime = 2
  %3 = arith.constant { problemStartTime = 3 } 3 : i32
  // ASAP-NEXT: asapStartTime = 3
  %4 = arith.constant { problemStartTime = 4 } 4 : i32
  // ASAP-NEXT: asapStartTime = 4
  %5 = arith.constant { problemStartTime = 5 } 5 : i32
  // ASAP-NEXT: asapStartTime = 5
  // SIMPLEX: return
  // SIMPLEX-SAME: simplexStartTime = 5
  return { problemStartTime = 6 }
}
