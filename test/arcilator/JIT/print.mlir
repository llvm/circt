// RUN: arcilator %s --run=basic | FileCheck %s
// REQUIRES: arcilator-jit

// CHECK: result = 4

func.func @basic() {
    %four = arith.constant 4 : i32
    arc.sim.emit "result", %four : i32
    return
}
