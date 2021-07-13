// RUN: circt-opt -test-asap-scheduler -verify-diagnostics -allow-unregistered-dialect -split-input-file %s

// Test basic functionality with a simple graph and unit latencies
func @test_asap1(%a1 : i32, %a2 : i32, %a3 : i32, %a4 : i32) -> i32 {
  // expected-remark@+1 {{start time = 0}}
  %0 = addi %a1, %a2 : i32
  // expected-remark@+1 {{start time = 1}}
  %1 = addi %0, %a3 : i32
  // expected-remark@+1 {{start time = 1}}
  %2 = addi %a4, %0 : i32
  // expected-remark@+1 {{start time = 2}}
  %3 = addi %2, %1 : i32
  // expected-remark@+1 {{start time = 3}}
  %4 = addi %3, %3 : i32
  // expected-remark@+1 {{start time = 4}}
  %5 = "more.operands"(%0, %1, %2, %3, %4) : (i32, i32, i32, i32, i32) -> i32
  // expected-remark@+1 {{start time = 5}}
  return %5 : i32
}

// -----

// Test non-unit latencies
func @test_asap2(%v : complex<f32>) -> f32 attributes { operatortypes = [
    { name = "extr", latency = 0 },
    { name = "add", latency = 3 },
    { name = "mult", latency = 6 },
    { name = "sqrt", latency = 10 }
  ] } {
  // expected-remark@+1 {{start time = 0}}
  %0 = "complex.re"(%v) { opr = "extr" } : (complex<f32>) -> f32
  // expected-remark@+1 {{start time = 0}}
  %1 = "complex.im"(%v) { opr = "extr" } : (complex<f32>) -> f32
  // expected-remark@+1 {{start time = 0}}
  %2 = mulf %0, %0 { opr = "mult" } : f32
  // expected-remark@+1 {{start time = 0}}
  %3 = mulf %1, %1 { opr = "mult" } : f32
  // expected-remark@+1 {{start time = 6}}
  %4 = addf %2, %3 { opr = "add" } : f32
  // expected-remark@+1 {{start time = 9}}
  %5 = "math.sqrt"(%4) { opr = "sqrt" } : (f32) -> f32
  // expected-remark@+1 {{start time = 19}}
  return %5 : f32
}

// -----

// Test auxiliary dependences
func @test_asap3() attributes { auxdeps = [
    [0,1], [0,2], [2,3], [3,4], [3,6], [4,5], [5,6]
  ] } {
  %0 = constant 0 : i32 // expected-remark {{start time = 0}}
  %1 = constant 1 : i32 // expected-remark {{start time = 1}}
  %2 = constant 2 : i32 // expected-remark {{start time = 1}}
  %3 = constant 3 : i32 // expected-remark {{start time = 2}}
  %4 = constant 4 : i32 // expected-remark {{start time = 3}}
  %5 = constant 5 : i32 // expected-remark {{start time = 4}}
  return // expected-remark {{start time = 5}}
}

// -----

// Test fail-safe for cyclic graphs
// expected-error@+2 {{dependence cycle detected}}
// expected-error@+1 {{scheduling failed}}
func @test_asap4() attributes { auxdeps = [
    [0,1], [1,2], [2,3], [3,1]
  ] } {
  %0 = constant 0 : i32
  %1 = constant 1 : i32
  %2 = constant 2 : i32
  %3 = constant 3 : i32
  return
}
