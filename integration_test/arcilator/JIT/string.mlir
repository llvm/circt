// RUN: arcilator %s --run | FileCheck --match-full-lines %s
// REQUIRES: arcilator-jit


func.func @entry() {
  %s1 = sim.string.literal "Hello"
  %l1 = sim.string.length %s1
  // CHECK: result = {{0*}}5
  arc.sim.emit "result", %l1 : i64
  %s2 = sim.string.literal "World"
  %l2 = sim.string.length %s2
  // CHECK: result = {{0*}}5
  arc.sim.emit "result", %l2 : i64
  %s3 = sim.string.concat (%s1, %s2)
  %l3 = sim.string.length %s3
  // CHECK: result = {{0*}}a
  arc.sim.emit "result", %l3 : i64
  return
}
