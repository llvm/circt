; RUN: firtool %s --preserve-aggregate=all --scalarize-top-module=false -ir-fir | FileCheck %s

FIRRTL version 3.0.0
circuit Top :
  type Passive = {a: UInt<1>}
  type NonPassive = {flip a : Passive }
  type ResetTy = Reset
  type UninferredWith = UInt
  ; CHECK-LABEL: firrtl.module @Top(in %in: !firrtl.alias<Passive, bundle<a: uint<1>>>, out %out_a: !firrtl.bundle<a: uint<1>>, out %o1: !firrtl.uint<1>, out %o2: !firrtl.uint<2>) {
  module Top :
    input in: Passive
    input out : NonPassive
    output o1: UninferredWith
    output o2: UninferredWith
    o1 <= UInt<1>(0)
    o2 <= UInt<2>(0)
    out.a <= in
