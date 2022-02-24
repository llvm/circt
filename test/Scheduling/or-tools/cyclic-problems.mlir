// REQUIRES: or-tools
// RUN: circt-opt %s -test-lp-scheduler=with=CyclicProblem | FileCheck %s -check-prefix=LP

// LP-LABEL: cyclic
// LP-SAME: lpInitiationInterval = 2
func @cyclic(%a1 : i32, %a2 : i32) -> i32 attributes {
  auxdeps = [ [4,1,1], [4,2,2] ],
  operatortypes = [ { name = "_0", latency = 0 }, { name = "_2", latency = 2 } ]
  } {
  %0 = arith.constant { problemStartTime = 0 } 42 : i32
  %1 = arith.addi %a1, %a2 { opr = "_0", problemStartTime = 2 } : i32
  %2 = arith.subi %a2, %a1 { opr = "_2", problemStartTime = 0 } : i32
  %3 = arith.muli %1, %2 { problemStartTime = 2 } : i32
  %4 = arith.divui %2, %0 { problemStartTime = 3 } : i32
  // LP: return
  // LP-SAME: lpStartTime = 3
  return %3 : i32
}

// LP-LABEL: mobility
// LP-SAME: lpInitiationInterval = 3
func @mobility() attributes {
  auxdeps = [
    [0,1], [0,2], [1,3], [2,3],
    [3,4], [3,5], [4,6], [5,6],
    [5,2,1]
  ],
  operatortypes = [ { name = "_4", latency = 4 }]
  } {
  %0 = arith.constant { problemStartTime = 0 } 0 : i32
  %1 = arith.constant { opr = "_4", problemStartTime = 1 } 1 : i32
  %2 = arith.constant { problemStartTime = 4 } 2 : i32
  %3 = arith.constant { problemStartTime = 5 } 3 : i32
  %4 = arith.constant { opr = "_4", problemStartTime = 6 } 4 : i32
  %5 = arith.constant { problemStartTime = 6} 5 : i32
  // LP: return
  // LP-SAME: lpStartTime = 10
  return
}

// LP-LABEL: interleaved_cycles
// LP-SAME: lpInitiationInterval = 4
func @interleaved_cycles() attributes {
  auxdeps = [
    [0,1], [0,2], [1,3], [2,3],
    [3,4], [4,7], [3,5], [5,6], [6,7],
    [7,8], [7,9], [8,10], [9,10],
    [6,2,2], [9,5,2]
  ],
  operatortypes = [ { name = "_10", latency = 10 } ]
  } {
  %0 = arith.constant { problemStartTime = 0 } 0 : i32
  %1 = arith.constant { opr = "_10", problemStartTime = 1 } 1 : i32
  %2 = arith.constant { problemStartTime = 10 } 2 : i32
  %3 = arith.constant { problemStartTime = 11 } 3 : i32
  %4 = arith.constant { opr = "_10", problemStartTime = 12 } 4 : i32
  %5 = arith.constant { problemStartTime = 16 } 5 : i32
  %6 = arith.constant { problemStartTime = 17 } 6 : i32
  %7 = arith.constant { problemStartTime = 22 } 7 : i32
  %8 = arith.constant { opr = "_10", problemStartTime = 23 } 8 : i32
  %9 = arith.constant { problemStartTime = 23 } 9 : i32
  // LP: return
  // LP-SAME: lpStartTime = 33
  return
}

// LP-LABEL: self_arc
// LP-SAME: lpInitiationInterval = 3
func @self_arc() -> i32 attributes {
  auxdeps = [ [1,1,1] ],
  operatortypes = [ { name = "_3", latency = 3 } ]
  } {
  %0 = arith.constant { problemStartTime = 0 } 1 : i32
  %1 = arith.muli %0, %0 { opr = "_3", problemStartTime = 1 } : i32
  // LP: return
  // LP-SAME: lpStartTime = 4
  return %1 : i32
}
