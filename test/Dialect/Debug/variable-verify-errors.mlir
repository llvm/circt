// RUN: circt-opt %s --verify-diagnostics --split-input-file

// A memory is never a port.

func.func @MemWithDir(%a: i8) {
  // expected-error @+1 {{of kind mem cannot have a direction}}
  dbg.variable "m", %a mem dir in : i8
  return
}

// -----

// A node is never a port.

func.func @NodeWithDir(%a: i8) {
  // expected-error @+1 {{of kind node cannot have a direction}}
  dbg.variable "n", %a node dir out : i8
  return
}

// -----

// A wire is never a port.

func.func @WireWithDir(%a: i8) {
  // expected-error @+1 {{of kind wire cannot have a direction}}
  dbg.variable "w", %a wire dir in : i8
  return
}

// -----

// A clock of builtin integer type is one bit wide.

func.func @ClockNotOneBit(%a: i8) {
  // expected-error @+1 {{of kind clock must be i1, got 'i8'}}
  %v = dbg.value %a clock : i8
  dbg.variable "clk", %v : !dbg.value
  return
}

// -----

func.func @ResetNotOneBit(%a: i8) {
  // expected-error @+1 {{of kind reset must be i1, got 'i8'}}
  %v = dbg.value %a reset : i8
  dbg.variable "rst", %v : !dbg.value
  return
}

// -----

func.func @AsyncResetNotOneBit(%a: i8) {
  // expected-error @+1 {{of kind asyncreset must be i1, got 'i8'}}
  %v = dbg.value %a asyncreset : i8
  dbg.variable "rst", %v : !dbg.value
  return
}
