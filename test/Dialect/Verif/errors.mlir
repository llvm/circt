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

// expected-error @below {{init region must have no arguments}}
verif.bmc bound 10 num_regs 0 init {
^bb0(%clk: !seq.clock):
} loop {
^bb0(%clk: !seq.clock):
} circuit {
^bb0(%clk: !seq.clock, %arg0: i32):
  %true = hw.constant true
  verif.assert %true : i1
  verif.yield %arg0 : i32
}

// -----

// expected-error @below {{init and loop regions must yield the same types of values}}
verif.bmc bound 10 num_regs 0 init {
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

// expected-error @below {{init and loop regions must yield the same types of values}}
verif.bmc bound 10 num_regs 0 init {
  %clkInit = hw.constant false
  %toClk = seq.to_clock %clkInit
  %c1_i2 = hw.constant 2 : i2
  verif.yield %toClk, %c1_i2 : !seq.clock, i2
} loop {
^bb0(%clk1: !seq.clock, %arg0: i32, %state: i1):
  verif.yield %clk1, %state : !seq.clock, i1
} circuit {
^bb0(%clk1: !seq.clock, %arg0: i32, %state: i1):
  %true = hw.constant true
  verif.assert %true : i1
  verif.yield %arg0 : i32
}

// -----

// expected-error @below {{loop region arguments must match the types of the values yielded by the init and loop regions}}
verif.bmc bound 10 num_regs 0 init {
  %clkInit = hw.constant false
  %toClk = seq.to_clock %clkInit
  %c1_i2 = hw.constant 2 : i2
  verif.yield %toClk, %c1_i2 : !seq.clock, i2
} loop {
^bb0(%clk1: !seq.clock, %state: i2, %arg0: i32):
  verif.yield %clk1, %state : !seq.clock, i2
} circuit {
^bb0(%clk1: !seq.clock, %arg0: i32, %state: i1):
  %true = hw.constant true
  verif.assert %true : i1
  verif.yield %arg0 : i32
}

// -----

// expected-error @below {{init and loop regions must yield at least as many clock values as there are clock arguments to the circuit region}}
verif.bmc bound 10 num_regs 0 init {
  %clkInit = hw.constant false
  %toClk = seq.to_clock %clkInit
  verif.yield %toClk: !seq.clock
} loop {
^bb0(%clk1: !seq.clock):
  verif.yield %clk1 : !seq.clock
} circuit {
^bb0(%clk1: !seq.clock, %clk2: !seq.clock, %arg0: i32):
  %true = hw.constant true
  verif.assert %true : i1
  verif.yield %arg0 : i32
}

// -----

// expected-error @below {{init and loop regions must yield as many clock values as there are clock arguments in the circuit region before any other values}}
verif.bmc bound 10 num_regs 0 init {
  %clkInit = hw.constant false
  %toClk = seq.to_clock %clkInit
  verif.yield %toClk, %clkInit, %toClk: !seq.clock, i1, !seq.clock
} loop {
^bb0(%clk1: !seq.clock, %state: i1, %clk2: !seq.clock):
  verif.yield %clk1, %state, %clk2 : !seq.clock, i1, !seq.clock
} circuit {
^bb0(%clk1: !seq.clock, %clk2: !seq.clock, %arg0: i32):
  %true = hw.constant true
  verif.assert %true : i1
  verif.yield %arg0 : i32
}
