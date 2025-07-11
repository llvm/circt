// RUN: circt-opt %s --convert-datapath-to-smt | FileCheck %s

// CHECK-LABEL: @compressor
hw.module @compressor(in %a : i4, in %b : i4, in %c : i4, out carry : i4, out save : i4) {
  // CHECK-NEXT: %[[C:.+]] = builtin.unrealized_conversion_cast %c : i4 to !smt.bv<4>
  // CHECK-NEXT: %[[B:.+]] = builtin.unrealized_conversion_cast %b : i4 to !smt.bv<4>
  // CHECK-NEXT: %[[A:.+]] = builtin.unrealized_conversion_cast %a : i4 to !smt.bv<4>
  // CHECK-NEXT: %[[AB:.+]] = smt.bv.add %[[A]], %[[B]] : !smt.bv<4>
  // CHECK-NEXT: %[[INS:.+]] = smt.bv.add %[[AB]], %[[C]] : !smt.bv<4>
  // CHECK-NEXT: %[[COMP0:.+]] = smt.declare_fun : !smt.bv<4>
  // CHECK-NEXT: %[[COMP0_BV:.+]] = builtin.unrealized_conversion_cast %[[COMP0]] : !smt.bv<4> to i4
  // CHECK-NEXT: %[[COMP1:.+]] = smt.declare_fun : !smt.bv<4>
  // CHECK-NEXT: %[[COMP1_BV:.+]] = builtin.unrealized_conversion_cast %7 : !smt.bv<4> to i4
  // CHECK-NEXT: %[[OUT:.+]] = smt.bv.add %[[COMP0]], %[[COMP1]] : !smt.bv<4>
  // CHECK-NEXT: %[[P:.+]] = smt.eq %[[INS]], %[[OUT]] : !smt.bv<4>  
  // CHECK-NEXT: smt.assert %[[P]]
  // CHECK-NEXT: hw.output %[[COMP0_BV]], %[[COMP1_BV]] : i4, i4
  %0:2 = datapath.compress %a, %b, %c : i4 [3 -> 2]
  hw.output %0#0, %0#1 : i4, i4
}

// CHECK-LABEL: @partial_product
hw.module @partial_product(in %a : i3, in %b : i3, out pp0 : i3, out pp1 : i3, out pp2 : i3) {
  //CHECK-NEXT: %[[B:.+]] = builtin.unrealized_conversion_cast %b : i3 to !smt.bv<3>
  //CHECK-NEXT: %[[A:.+]] = builtin.unrealized_conversion_cast %a : i3 to !smt.bv<3>
  //CHECK-NEXT: %[[MUL:.+]] = smt.bv.mul %[[A]], %[[B]] : !smt.bv<3>
  //CHECK-NEXT: %[[PP0:.+]] = smt.declare_fun : !smt.bv<3>
  //CHECK-NEXT: %[[PP0_BV:.+]] = builtin.unrealized_conversion_cast %[[PP0]] : !smt.bv<3> to i3
  //CHECK-NEXT: %[[PP1:.+]] = smt.declare_fun : !smt.bv<3>
  //CHECK-NEXT: %[[PP1_BV:.+]] = builtin.unrealized_conversion_cast %[[PP1]] : !smt.bv<3> to i3
  //CHECK-NEXT: %[[PP2:.+]] = smt.declare_fun : !smt.bv<3>
  //CHECK-NEXT: %[[PP2_BV:.+]] = builtin.unrealized_conversion_cast %[[PP2]] : !smt.bv<3> to i3
  //CHECK-NEXT: %[[ADD01:.+]] = smt.bv.add %[[PP0]], %[[PP1]] : !smt.bv<3>
  //CHECK-NEXT: %[[ADD012:.+]] = smt.bv.add %[[ADD01]], %[[PP2]] : !smt.bv<3>
  //CHECK-NEXT: %[[P:.+]] = smt.eq %[[MUL]], %[[ADD012]] : !smt.bv<3>
  //CHECK-NEXT: smt.assert %[[P]]
  //CHECK-NEXT: hw.output %[[PP0_BV]], %[[PP1_BV]], %[[PP2_BV]] : i3, i3, i3
  %0:3 = datapath.partial_product %a, %b : (i3, i3) -> (i3, i3, i3)
  hw.output %0#0, %0#1, %0#2 : i3, i3, i3
}
