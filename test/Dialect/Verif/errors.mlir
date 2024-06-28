// RUN: circt-opt %s --split-input-file --verify-diagnostics

// expected-error @below {{types of the yielded values of both regions must match}}
verif.lec first {
^bb0(%arg0: i32):
  verif.yield %arg0 : i32
} second {
^bb0(%arg0: i32):
  verif.yield
}

// -----

// expected-error @below {{block argument types of both regions must match}}
verif.lec first {
^bb0(%arg0: i32, %arg1: i32):
  verif.yield %arg0 : i32
} second {
^bb0(%arg0: i32):
  verif.yield %arg0 : i32
}

// -----

// expected-error @below {{op block argument types of loop and circuit regions must match}}
verif.bmc bound 10 attributes {verif.some_attr} init {
} loop {
} circuit {
^bb0(%arg0: i32):
  %true = hw.constant true
  // Arbitrary assertion to avoid failure
  verif.assert %true : i1
  verif.yield %arg0 : i32
}

// -----

// expected-error @below {{number of yielded values in init and loop regions must match the number of clock inputs in the circuit region}}
verif.bmc bound 10 attributes {verif.some_attr} init {
} loop {
^bb0(%clk: !seq.clock, %arg0: i32):
} circuit {
^bb0(%clk: !seq.clock, %arg0: i32):
  %true = hw.constant true
  verif.assert %true : i1
  verif.yield %arg0 : i32
}

// -----

// expected-error @below {{init region must have no arguments}}
verif.bmc bound 10 attributes {verif.some_attr} init {
^bb0(%clk: !seq.clock, %arg0: i32):
} loop {
^bb0(%clk: !seq.clock, %arg0: i32):
} circuit {
^bb0(%clk: !seq.clock, %arg0: i32):
  %true = hw.constant true
  verif.assert %true : i1
  verif.yield %arg0 : i32
}

// -----

// expected-error @below {{init region must only yield clock values}}
verif.bmc bound 10 attributes {verif.some_attr} init {
  %clkInit = hw.constant false
  %toClk = seq.to_clock %clkInit
  verif.yield %toClk, %clkInit : !seq.clock, i1
} loop {
^bb0(%clk1: !seq.clock, %clk2: !seq.clock, %arg0: i32):
  verif.yield %clk1, %clk2 : !seq.clock, !seq.clock
} circuit {
^bb0(%clk1: !seq.clock, %clk2: !seq.clock, %arg0: i32):
  %true = hw.constant true
  verif.assert %true : i1
  verif.yield %arg0 : i32
}

// -----

// expected-error @below {{loop region must only yield clock values}}
verif.bmc bound 10 attributes {verif.some_attr} init {
  %clkInit = hw.constant false
  %toClk = seq.to_clock %clkInit
  verif.yield %toClk, %toClk : !seq.clock, !seq.clock
} loop {
^bb0(%clk1: !seq.clock, %clk2: !seq.clock, %arg0: i32):
  verif.yield %clk1, %arg0 : !seq.clock, i32
} circuit {
^bb0(%clk1: !seq.clock, %clk2: !seq.clock, %arg0: i32):
  %true = hw.constant true
  verif.assert %true : i1
  verif.yield %arg0 : i32
}

// -----

// expected-error @below {{op block argument types of loop and circuit regions must match}}
verif.bmc bound 10 attributes {verif.some_attr} init {
} loop {
} circuit {
^bb0(%arg0: i32):
  verif.yield %arg0 : i32
}
