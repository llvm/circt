// RUN: circt-opt -test-asap-scheduler -verify-diagnostics -allow-unregistered-dialect %s

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
