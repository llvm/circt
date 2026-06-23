// RUN: arcilator --run %s --jit-entry=entry | FileCheck --match-full-lines %s
// REQUIRES: arcilator-jit

func.func @entry() {
  %three = arith.constant 3.0 : f64
  %four = arith.constant 4.0 : f64
  %five = arith.constant 5.0 : f64

  %hypot = sim.real.hypot %three, %four : f64
  %ok = arith.cmpf oeq, %hypot, %five : f64

  // CHECK: hypot_ok = 1
  arc.sim.emit "hypot_ok", %ok : i1

  return
}
