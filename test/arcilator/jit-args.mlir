// RUN: arcilator --run %s --jit-entry=entry --args=0x10,0x1234567890ABCDEF1234567890ABC  | FileCheck %s

module {
  func.func @entry(%arg0: i32, %arg1: i116) {
    // CHECK: arg0 = 00000010
    arc.sim.emit "arg0", %arg0 : i32
    // CHECK: arg1 = 1234567890abcdef1234567890abc
    arc.sim.emit "arg1", %arg1 : i116
    return
  }
}
