// RUN: circt-opt %s -test-chaining-problem -allow-unregistered-dialect
// RUN: circt-opt %s -test-simplex-scheduler=with=ChainingProblem -allow-unregistered-dialect | FileCheck %s -check-prefix=SIMPLEX

// SIMPLEX-LABEL: adder_chain
func @adder_chain(%arg0 : i32, %arg1 : i32) -> i32 attributes {
  cycletime = 5.0, // only evaluated for scheduler test; ignored by the problem test!
  operatortypes = [
   { name = "add", latency = 0, incdelay = 2.34, outdelay = 2.34}
  ] } {
  %0 = arith.addi %arg0, %arg1 { opr = "add", problemStartTime = 0, problemStartTimeInCycle = 0.0 } : i32
  %1 = arith.addi %0, %arg1 { opr = "add", problemStartTime = 0, problemStartTimeInCycle = 2.34 } : i32
  %2 = arith.addi %1, %arg1 { opr = "add", problemStartTime = 0, problemStartTimeInCycle = 4.68 } : i32
  %3 = arith.addi %2, %arg1 { opr = "add", problemStartTime = 1, problemStartTimeInCycle = 0.0 } : i32
  %4 = arith.addi %3, %arg1 { opr = "add", problemStartTime = 1, problemStartTimeInCycle = 2.34 } : i32
  // SIMPLEX: return
  // SIMPLEX-SAME: simplexStartTime = 2
  return { problemStartTime = 2, problemStartTimeInCycle = 0.0 } %4 : i32
}

// SIMPLEX-LABEL: multi_cycle
func @multi_cycle(%arg0 : i32, %arg1 : i32) -> i32 attributes {
  cycletime = 5.0, // only evaluated for scheduler test; ignored by the problem test!
  operatortypes = [
   { name = "add", latency = 0, incdelay = 2.34, outdelay = 2.34},
   { name = "mul", latency = 3, incdelay = 2.5, outdelay = 3.75}
  ] } {
  %0 = arith.addi %arg0, %arg1 { opr = "add", problemStartTime = 0, problemStartTimeInCycle = 0.0 } : i32
  %1 = arith.addi %0, %arg1 { opr = "add", problemStartTime = 0, problemStartTimeInCycle = 2.34 } : i32
  %2 = arith.muli %1, %0 { opr = "mul", problemStartTime = 0, problemStartTimeInCycle = 4.68 } : i32
  %3 = arith.addi %2, %1 { opr = "add", problemStartTime = 3, problemStartTimeInCycle = 3.75 } : i32
  %4 = arith.addi %3, %2 { opr = "add", problemStartTime = 3, problemStartTimeInCycle = 6.09 } : i32
  // SIMPLEX: return
  // SIMPLEX-SAME: simplexStartTime = 5
  return { problemStartTime = 4, problemStartTimeInCycle = 0.0 } %4 : i32
}
