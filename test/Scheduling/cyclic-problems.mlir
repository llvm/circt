// RUN: circt-opt %s -test-cyclic-problem

func @cyclic(%a1 : i32, %a2 : i32) -> i32 attributes {
  problemInitiationInterval = 2,
  auxdeps = [ [4,1,1], [4,2,2] ],
  operatortypes = [ { name = "_0", latency = 0 }, { name = "_2", latency = 2 } ]
  } {
  %c42 = constant { problemStartTime = 0 } 42 : i32
  %0 = addi %a1, %a2 { opr = "_0", problemStartTime = 2 } : i32
  %1 = subi %a2, %a1 { opr = "_2", problemStartTime = 0 } : i32
  %2 = muli %0, %1 { problemStartTime = 2 } : i32
  %3 = divi_unsigned %1, %c42 { problemStartTime = 3 } : i32
  return { problemStartTime = 4 } %2 : i32
}
