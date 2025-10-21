// RUN: arcilator %s --run | FileCheck %s
// REQUIRES: arcilator-jit

// CHECK: result = {{0*}}4

func.func @entry() {
  %four = arith.constant 4 : i32
  arc.sim.emit "result", %four : i32
  return
}
