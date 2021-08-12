// RUN: circt-opt %s -test-cyclic-problem
// RUN: circt-opt %s -test-simplex-scheduler | FileCheck %s -check-prefix=SIMPLEX

// SIMPLEX-LABEL: cyclic
// SIMPLEX-SAME: simplexInitiationInterval = 2
func @cyclic(%a1 : i32, %a2 : i32) -> i32 attributes {
  problemInitiationInterval = 2,
  auxdeps = [ [4,1,1], [4,2,2] ],
  operatortypes = [ { name = "_0", latency = 0 }, { name = "_2", latency = 2 } ]
  } {
  // SIMPLEX-NEXT: simplexStartTime = 0
  %0 = constant { problemStartTime = 0 } 42 : i32
  // SIMPLEX-NEXT: simplexStartTime = 1
  %1 = addi %a1, %a2 { opr = "_0", problemStartTime = 2 } : i32
  // SIMPLEX-NEXT: simplexStartTime = 0
  %2 = subi %a2, %a1 { opr = "_2", problemStartTime = 0 } : i32
  // SIMPLEX-NEXT: simplexStartTime = 2
  %3 = muli %1, %2 { problemStartTime = 2 } : i32
  // SIMPLEX-NEXT: simplexStartTime = 2
  %4 = divi_unsigned %2, %0 { problemStartTime = 3 } : i32
  // SIMPLEX-NEXT: simplexStartTime = 3
  return { problemStartTime = 4 } %3 : i32
}

// SIMPLEX-LABEL: mobility
// SIMPLEX-SAME: simplexInitiationInterval = 3
func @mobility() attributes {
  problemInitiationInterval = 3,
  auxdeps = [
    [0,1], [0,2], [1,3], [2,3],
    [3,4], [3,5], [4,6], [5,6],
    [5,2,1]
  ],
  operatortypes = [ { name = "_4", latency = 4 }]
  } {
  // SIMPLEX-NEXT: simplexStartTime = 0
  %0 = constant { problemStartTime = 0 } 0 : i32
  // SIMPLEX-NEXT: simplexStartTime = 1
  %1 = constant { opr = "_4", problemStartTime = 1 } 1 : i32
  // SIMPLEX-NEXT: simplexStartTime = 4
  %2 = constant { problemStartTime = 4 } 2 : i32
  // SIMPLEX-NEXT: simplexStartTime = 5
  %3 = constant { problemStartTime = 5 } 3 : i32
  // SIMPLEX-NEXT: simplexStartTime = 6
  %4 = constant { opr = "_4", problemStartTime = 6 } 4 : i32
  // SIMPLEX-NEXT: simplexStartTime = 6
  %5 = constant { problemStartTime = 6} 5 : i32
  // SIMPLEX-NEXT: simplexStartTime = 10
  return { problemStartTime = 10 }
}

// SIMPLEX-LABEL: interleaved_cycles
// SIMPLEX-SAME: simplexInitiationInterval = 4
func @interleaved_cycles() attributes {
  problemInitiationInterval = 4,
  auxdeps = [
    [0,1], [0,2], [1,3], [2,3],
    [3,4], [4,7], [3,5], [5,6], [6,7],
    [7,8], [7,9], [8,10], [9,10],
    [6,2,2], [9,5,2]
  ],
  operatortypes = [ { name = "_10", latency = 10 } ]
  } {
  // SIMPLEX-NEXT: simplexStartTime = 0
  %0 = constant { problemStartTime = 0 } 0 : i32
  // SIMPLEX-NEXT: simplexStartTime = 1
  %1 = constant { opr = "_10", problemStartTime = 1 } 1 : i32
  // SIMPLEX-NEXT: simplexStartTime = 10
  %2 = constant { problemStartTime = 10 } 2 : i32
  // SIMPLEX-NEXT: simplexStartTime = 11
  %3 = constant { problemStartTime = 11 } 3 : i32
  // SIMPLEX-NEXT: simplexStartTime = 12
  %4 = constant { opr = "_10", problemStartTime = 12 } 4 : i32
  // SIMPLEX-NEXT: simplexStartTime = 16
  %5 = constant { problemStartTime = 16 } 5 : i32
  // SIMPLEX-NEXT: simplexStartTime = 17
  %6 = constant { problemStartTime = 17 } 6 : i32
  // SIMPLEX-NEXT: simplexStartTime = 22
  %7 = constant { problemStartTime = 22 } 7 : i32
  // SIMPLEX-NEXT: simplexStartTime = 23
  %8 = constant { opr = "_10", problemStartTime = 23 } 8 : i32
  // SIMPLEX-NEXT: simplexStartTime = 23
  %9 = constant { problemStartTime = 23 } 9 : i32
  // SIMPLEX-NEXT: simplexStartTime = 33
  return { problemStartTime = 33 }
}
