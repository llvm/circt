; RUN: firtool %s --preserve-aggregate=all --scalarize-top-module=false -ir-fir | FileCheck %s

FIRRTL version 3.0.0
circuit Top :
  type Passive = {a: UInt<1>}
  type NonPassive = {flip a : Passive }
  type UInt2 = UInt<2>
  type UInt3 = UInt<2>
  type ResetTy = Reset
  type UninferredWith = UInt
  ; CHECK-LABEL: firrtl.module @Top(
  ; CHECK-SAME: in %in: !firrtl.alias<Passive, bundle<a: uint<1>>>
  ; CHECK-SAME: in %u2: !firrtl.alias<UInt2, uint<2>>
  ; CHECK-SAME: out %out_a: !firrtl.bundle<a: uint<1>>
  ; CHECK-SAME: out %o1: !firrtl.uint<1>
  ; CHECK-SAME: out %o2: !firrtl.uint<3>
  ; CHECK-SAME: out %o3: !firrtl.uint<3>
  module Top :
    input in: Passive
    input u2: UInt2
    input out : NonPassive
    output o1: UninferredWith
    output o2: UninferredWith
    output o3: UninferredWith
    o1 <= UInt<1>(0)
    o2 <= add(u2, u2)
    o3 <= sub(u2, u2)
    out.a <= in
