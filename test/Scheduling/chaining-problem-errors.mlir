// RUN: circt-opt %s -test-chaining-problem -verify-diagnostics -split-input-file

// expected-error@+2 {{Invalid cycle time}}
// expected-error@+1 {{problem check failed}}
func @invalid_cycletime() attributes {
  cycletime = -3.14
  } {
  return
}

// -----

// expected-error@+2 {{Missing delays}}
// expected-error@+1 {{problem check failed}}
func @missing_delay() attributes {
  cycletime = 10.0, operatortypes = [
    { name = "foo", latency = 0}
  ]} {
  return
}

// -----

// expected-error@+2 {{Negative delays}}
// expected-error@+1 {{problem check failed}}
func @negative_delay() attributes {
  cycletime = 10.0, operatortypes = [
    { name = "foo", latency = 0, incdelay = -1.0, outdelay = -1.0}
  ]} {
  return
}

// -----

// expected-error@+2 {{Incoming delay (2.000000e+00) for operator type 'foo' exceeds cycle time}}
// expected-error@+1 {{problem check failed}}
func @incdelay_exceeds() attributes {
  cycletime = 1.0, operatortypes = [
    { name = "foo", latency = 1, incdelay = 2.0, outdelay = 1.0}
  ]} {
  return
}

// -----

// expected-error@+2 {{Outgoing delay (2.000000e+00) for operator type 'foo' exceeds cycle time}}
// expected-error@+1 {{problem check failed}}
func @outdelay_exceeds() attributes {
  cycletime = 1.0, operatortypes = [
    { name = "foo", latency = 1, incdelay = 1.0, outdelay = 2.0}
  ]} {
  return
}

// -----

// expected-error@+2 {{Incoming & outgoing delay must be equal for zero-latency operator type}}
// expected-error@+1 {{problem check failed}}
func @inc_out_mismatch() attributes {
  cycletime = 10.0, operatortypes = [
    { name = "foo", latency = 0, incdelay = 1.0, outdelay = 2.0}
  ]} {
  return
}

// -----

// expected-error@+1 {{problem verification failed}}
func @no_pst() attributes { cycletime = 10.0} {
  // expected-error@+1 {{Operation has no start time in cycle}}
  return { problemStartTime = 0 }
}

// -----

// expected-error@+1 {{problem verification failed}}
func @cycle_time_exceeded() attributes {
  cycletime = 10.0, operatortypes = [
    { name = "foo", latency = 0, incdelay = 1.0, outdelay = 1.0}
  ] } {
  // expected-error@+1 {{Operation violates cycle time constraint}}
  return { opr = "foo", problemStartTime = 0, problemStartTimeInCycle = 9.5 }
}

// -----

// expected-error@+2 {{Precedence violated in cycle 0}}
// expected-error@+1 {{problem verification failed}}
func @precedence1(%arg0 : i32, %arg1 : i32) attributes {
  cycletime = 10.0, operatortypes = [
     { name = "add", latency = 0, incdelay = 1.0, outdelay = 1.0}
    ] } {
    %0 = arith.addi %arg0, %arg1 { opr = "add", problemStartTime = 0, problemStartTimeInCycle = 1.1 } : i32
    %1 = arith.addi %0, %arg1 { opr = "add", problemStartTime = 0, problemStartTimeInCycle = 2.0 } : i32
    return { problemStartTime = 0, problemStartTimeInCycle = 0.0 }
}

// -----

// expected-error@+2 {{Precedence violated in cycle 3}}
// expected-error@+1 {{problem verification failed}}
func @precedence2(%arg0 : i32, %arg1 : i32) attributes {
  cycletime = 10.0, operatortypes = [
     { name = "add", latency = 0, incdelay = 1.0, outdelay = 1.0},
     { name = "mul", latency = 3, incdelay = 2.5, outdelay = 3.75}
    ] } {
    %0 = arith.muli %arg0, %arg1 { opr = "mul", problemStartTime = 0, problemStartTimeInCycle = 0.0 } : i32
    %1 = arith.addi %0, %arg1 { opr = "add", problemStartTime = 3, problemStartTimeInCycle = 3.0 } : i32
    return { problemStartTime = 4, problemStartTimeInCycle = 0.0 }
}
