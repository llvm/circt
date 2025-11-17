// RUN: arcilator --run %s --jit-entry=entry --args=1,2 --args=3,4 | FileCheck %s

module {
  func.func @entry(%arg0: i32, %arg1: i32) {
    // CHECK: arg0 = 00000001
    // CHECK: arg1 = 00000002
    // CHECK: arg0 = 00000003
    // CHECK: arg1 = 00000004
    arc.sim.emit "arg0", %arg0 : i32
    arc.sim.emit "arg1", %arg1 : i32
    return
  }
}
