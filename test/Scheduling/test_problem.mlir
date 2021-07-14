// RUN: circt-opt -test-scheduling-problem -allow-unregistered-dialect %s

// Test verification with a simple graph, unit latencies and an ASAP schedule
func @test_prob2a(%a1 : i32, %a2 : i32, %a3 : i32, %a4 : i32) -> i32 {
  %0 = addi %a1, %a2 {startTime = 0} : i32
  %1 = addi %0, %a3 {startTime = 1} : i32
  %2 = addi %a4, %0 {startTime = 1} : i32
  %3 = addi %2, %1 {startTime = 2} : i32
  %4 = addi %3, %3 {startTime = 3} : i32
  %5 = "more.operands"(%0, %1, %2, %3, %4) {startTime = 4} : (i32, i32, i32, i32, i32) -> i32
  return {startTime = 5} %5 : i32
}

// Test verification with an arbitrary but valid schedule
func @test_prob2c(%a1 : i32, %a2 : i32, %a3 : i32, %a4 : i32) -> i32 {
  %0 = addi %a1, %a2 {startTime = 3} : i32
  %1 = addi %0, %a3 {startTime = 7} : i32
  %2 = addi %a4, %0 {startTime = 19} : i32
  %3 = addi %2, %1 {startTime = 21} : i32
  %4 = addi %3, %3 {startTime = 42} : i32
  %5 = "more.operands"(%0, %1, %2, %3, %4) {startTime = 123} : (i32, i32, i32, i32, i32) -> i32
  return {startTime = 1000} %5 : i32
}

// Test verification with non-unit latencies and an ASAP schedule
func @test_prob3a(%v : complex<f32>) -> f32 attributes { operatortypes = [
    { name = "extr", latency = 0 },
    { name = "add", latency = 3 },
    { name = "mult", latency = 6 },
    { name = "sqrt", latency = 10 }
  ] } {
  %0 = "complex.re"(%v) { opr = "extr", startTime = 0 } : (complex<f32>) -> f32
  %1 = "complex.im"(%v) { opr = "extr", startTime = 0 } : (complex<f32>) -> f32
  %2 = mulf %0, %0 { opr = "mult", startTime = 0 } : f32
  %3 = mulf %1, %1 { opr = "mult", startTime = 0 } : f32
  %4 = addf %2, %3 { opr = "add", startTime = 6 } : f32
  %5 = "math.sqrt"(%4) { opr = "sqrt", startTime = 9} : (f32) -> f32
  return { startTime = 19 } %5 : f32
}


// Test verification of auxiliary dependences with an ASAP schedule
func @test_prob4a() attributes { auxdeps = [
    [0,1], [0,2], [2,3], [3,4], [3,6], [4,5], [5,6]
  ] } {
  %0 = constant { startTime = 0 } 0 : i32
  %1 = constant { startTime = 1 } 1 : i32
  %2 = constant { startTime = 1 } 2 : i32
  %3 = constant { startTime = 2 } 3 : i32
  %4 = constant { startTime = 3 } 4 : i32
  %5 = constant { startTime = 4 } 5 : i32
  return { startTime = 5 }
}
