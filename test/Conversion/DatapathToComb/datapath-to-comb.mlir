// RUN: circt-opt %s --convert-datapath-to-comb | FileCheck %s

// CHECK-LABEL: @compressor
hw.module @compressor(in %a : i2, in %b : i2, in %c : i2, out carry : i2, out save : i2) {
  //CHECK-NEXT: %[[A0:.+]] = comb.extract %a from 0 : (i2) -> i1
  //CHECK-NEXT: %[[A1:.+]] = comb.extract %a from 1 : (i2) -> i1
  //CHECK-NEXT: %[[B0:.+]] = comb.extract %b from 0 : (i2) -> i1
  //CHECK-NEXT: %[[B1:.+]] = comb.extract %b from 1 : (i2) -> i1
  //CHECK-NEXT: %[[C0:.+]] = comb.extract %c from 0 : (i2) -> i1
  //CHECK-NEXT: %[[C1:.+]] = comb.extract %c from 1 : (i2) -> i1
  //CHECK-NEXT: %false = hw.constant false
  //CHECK-NEXT: %[[AxB0:.+]] = comb.xor bin %[[A0]], %[[B0]] : i1
  //CHECK-NEXT: %[[AxBxC0:.+]] = comb.xor bin %[[AxB0]], %[[C0]] : i1
  //CHECK-NEXT: %[[AB0:.+]] = comb.and bin %[[A0]], %[[B0]] : i1
  //CHECK-NEXT: %[[AxBC0:.+]] = comb.and bin %[[AxB0]], %[[C0]] : i1
  //CHECK-NEXT: %[[AB0oAxBC0:.+]] = comb.or bin %[[AB0]], %[[AxBC0]] : i1
  //CHECK-NEXT: %[[AxB1:.+]] = comb.xor bin %[[A1]], %[[B1]] : i1
  //CHECK-NEXT: %[[AxBxC1:.+]] = comb.xor bin %[[AxB1]], %[[C1]] : i1
  //CHECK-NEXT: %[[AB1:.+]] = comb.and bin %[[A1]], %[[B1]] : i1
  //CHECK-NEXT: %[[AxBC1:.+]] = comb.and bin %[[AxB1]], %[[C1]] : i1
  //CHECK-NEXT: comb.or bin %[[AB1]], %[[AxBC1]] : i1
  //CHECK-NEXT: comb.concat %[[AxBxC1]], %[[AxBxC0]] : i1, i1
  //CHECK-NEXT: comb.concat %[[AB0oAxBC0]], %false : i1, i1
  %0:2 = datapath.compress %a, %b, %c : i2 [3 -> 2]
  hw.output %0#0, %0#1 : i2, i2
}

// CHECK-LABEL: @partial_product
hw.module @partial_product(in %a : i3, in %b : i3, out pp0 : i3, out pp1 : i3, out pp2 : i3) {
  // CHECK-NEXT: %[[B0:.+]] = comb.extract %b from 0 : (i3) -> i1
  // CHECK-NEXT: %[[B1:.+]] = comb.extract %b from 1 : (i3) -> i1
  // CHECK-NEXT: %[[B2:.+]] = comb.extract %b from 2 : (i3) -> i1
  // CHECK-NEXT: %[[B0R:.+]] = comb.replicate %[[B0]] : (i1) -> i3
  // CHECK-NEXT: %[[PP0:.+]] = comb.and %[[B0R]], %a : i3
  // CHECK-NEXT: %c0_i3 = hw.constant 0 : i3
  // CHECK-NEXT: comb.shl %[[PP0]], %c0_i3 : i3
  // CHECK-NEXT: %[[B1R:.+]] = comb.replicate %[[B1]] : (i1) -> i3
  // CHECK-NEXT: %[[PP1:.+]]  = comb.and %[[B1R]], %a : i3
  // CHECK-NEXT: %c1_i3 = hw.constant 1 : i3
  // CHECK-NEXT: comb.shl %[[PP1]], %c1_i3 : i3
  // CHECK-NEXT: %[[B2R:.+]] = comb.replicate %[[B2]] : (i1) -> i3
  // CHECK-NEXT: %[[PP2:.+]] = comb.and %[[B2R]], %a : i3
  // CHECK-NEXT: %c2_i3 = hw.constant 2 : i3
  // CHECK-NEXT: comb.shl %[[PP2]], %c2_i3 : i3
  %0:3 = datapath.partial_product %a, %b : (i3, i3) -> (i3, i3, i3)
  hw.output %0#0, %0#1, %0#2 : i3, i3, i3
}
