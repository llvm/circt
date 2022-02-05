// RUN: circt-opt %s -test-simplex-scheduler=with=ChainingProblem -verify-diagnostics -split-input-file

// expected-error@+2 {{Delays of operator type 'inv' exceed maximum cycle time}}
// expected-error@+1 {{scheduling failed}}
func @invalid_delay() attributes {
  cycletime = 2.0,  operatortypes = [{ name = "inv", latency = 0, incdelay = 2.34, outdelay = 2.34}] } {
  return { opr = "inv" }
}
