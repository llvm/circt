// REQUIRES: or-tools
// RUN: circt-opt %s -test-lp-scheduler=with=Problem -allow-unregistered-dialect | FileCheck %s -check-prefix=LP

// LP-LABEL: unit_latencies
func @unit_latencies(%a1 : i32, %a2 : i32, %a3 : i32, %a4 : i32) -> i32 {
  %0 = arith.addi %a1, %a2 : i32
  %1 = arith.addi %0, %a3 : i32
  %2:3 = "more.results"(%0, %1) : (i32, i32) -> (i32, i32, i32)
  %3 = arith.addi %a4, %2#1 : i32
  %4 = arith.addi %2#0, %2#2 : i32
  %5 = arith.addi %3, %3 : i32
  %6 = "more.operands"(%3, %4, %5) : (i32, i32, i32) -> i32
  // LP: return
  // LP-SAME: lpStartTime = 6
  return %6 : i32
}

// LP-LABEL: arbitrary_latencies
func @arbitrary_latencies(%v : complex<f32>) -> f32 attributes {
  operatortypes = [
    { name = "extr", latency = 0 },
    { name = "add", latency = 3 },
    { name = "mult", latency = 6 },
    { name = "sqrt", latency = 10 }
  ] } {
  %0 = "complex.re"(%v) { opr = "extr" } : (complex<f32>) -> f32
  %1 = "complex.im"(%v) { opr = "extr" } : (complex<f32>) -> f32
  %2 = arith.mulf %0, %0 { opr = "mult" } : f32
  %3 = arith.mulf %1, %1 { opr = "mult" } : f32
  %4 = arith.addf %2, %3 { opr = "add" } : f32
  %5 = "math.sqrt"(%4) { opr = "sqrt" } : (f32) -> f32
  // LP: return
  // LP-SAME: lpStartTime = 19
  return %5 : f32
}

// LP-LABEL: auxiliary_dependences
func @auxiliary_dependences() attributes { auxdeps = [
    [0,1], [0,2], [2,3], [3,4], [3,6], [4,5], [5,6]
  ] } {
  %0 = arith.constant 0 : i32
  %1 = arith.constant 1 : i32
  %2 = arith.constant 2 : i32
  %3 = arith.constant 3 : i32
  %4 = arith.constant 4 : i32
  %5 = arith.constant 5 : i32
  // LP: return
  // LP-SAME: lpStartTime = 5
  return { problemStartTime = 6 }
}
