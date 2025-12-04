// RUN: arcilator --run %s | FileCheck --match-full-lines %s
// REQUIRES: arcilator-jit

module {
  func.func @entry() {
    // CHECK: a1 = 1
    %a1 = arith.constant 0x1 : i4
    arc.sim.emit "a1", %a1 : i4

    // CHECK: v1 = 0001
    %v1 = arith.constant 0x1 : i16
    arc.sim.emit "v1", %v1 : i16

    // CHECK: v2 = abcdef
    %v2 = arith.constant 0xABCDEF : i24
    arc.sim.emit "v2", %v2 : i24

    // CHECK: v3 = 0123456789abcdef
    %v3 = arith.constant 0x0123456789ABCDEF : i64
    arc.sim.emit "v3", %v3 : i64

    // CHECK: v4 = 0000000000000001
    %v4 = arith.constant 1 : i64
    arc.sim.emit "v4", %v4 : i64

    // CHECK: v5 = 7a
    %v5 = arith.constant 0x7A : i7
    arc.sim.emit "v5", %v5 : i7

    // CHECK: v6 = 10000000000b
    %v6 = arith.constant 0x10000000000B : i47
    arc.sim.emit "v6", %v6 : i47

    // CHECK: v7 = 1
    %v7 = arith.constant 0x1 : i1
    arc.sim.emit "v7", %v7 : i1

    // CHECK: v8 = 0fde6741e3a44d997e80d393ae643ee0fd9
    %v8 = arith.constant 0xfde6741e3a44d997e80d393ae643ee0fd9 : i137
    arc.sim.emit "v8", %v8 : i137

    return
  }
}
