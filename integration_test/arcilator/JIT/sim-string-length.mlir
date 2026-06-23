// RUN: arcilator --run %s --jit-entry=entry | FileCheck --match-full-lines %s
// REQUIRES: arcilator-jit

// Exercise sim.string.length lowering through the sim string runtime bound into
// the JIT.

func.func @entry() {
  %str = sim.string.literal "hello"
  %len64 = sim.string.length %str
  %len = arith.trunci %len64 : i64 to i32
  // CHECK: len = 00000005
  arc.sim.emit "len", %len : i32
  return
}
