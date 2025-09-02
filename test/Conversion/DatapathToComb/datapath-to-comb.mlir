// RUN: circt-opt %s --pass-pipeline="builtin.module(hw.module(convert-datapath-to-comb))" | FileCheck %s
// RUN: circt-opt %s --pass-pipeline="builtin.module(hw.module(convert-datapath-to-comb{lower-compress-to-add=true}))" | FileCheck %s --check-prefix=TO-ADD
// RUN: circt-opt %s --pass-pipeline="builtin.module(hw.module(convert-datapath-to-comb{lower-partial-product-to-booth=true}, canonicalize))" | FileCheck %s --check-prefix=FORCE-BOOTH
// RUN: circt-opt %s --pass-pipeline="builtin.module(hw.module(convert-datapath-to-comb{timing-aware=true}))" | FileCheck %s --check-prefix=TIMING

// CHECK-LABEL: @compressor
hw.module @compressor(in %a : i2, in %b : i2, in %c : i2, out carry : i2, out save : i2) {
  //CHECK-NEXT: %false = hw.constant false
  //CHECK-NEXT: %[[A0:.+]] = comb.extract %a from 0 : (i2) -> i1
  //CHECK-NEXT: %[[A1:.+]] = comb.extract %a from 1 : (i2) -> i1
  //CHECK-NEXT: %[[B0:.+]] = comb.extract %b from 0 : (i2) -> i1
  //CHECK-NEXT: %[[B1:.+]] = comb.extract %b from 1 : (i2) -> i1
  //CHECK-NEXT: %[[C0:.+]] = comb.extract %c from 0 : (i2) -> i1
  //CHECK-NEXT: %[[C1:.+]] = comb.extract %c from 1 : (i2) -> i1
  //CHECK-NEXT: %[[AxB0:.+]] = comb.xor bin %[[A0]], %[[B0]] : i1
  //CHECK-NEXT: %[[AxBxC0:.+]] = comb.xor bin %[[AxB0]], %[[C0]] : i1
  //CHECK-NEXT: %[[AB0:.+]] = comb.and bin %[[A0]], %[[B0]] : i1
  //CHECK-NEXT: %[[AxBC0:.+]] = comb.and bin %[[AxB0]], %[[C0]] : i1
  //CHECK-NEXT: %[[AB0oAxBC0:.+]] = comb.or bin %[[AB0]], %[[AxBC0]] : i1
  //CHECK-NEXT: %[[AxB1:.+]] = comb.xor bin %[[A1]], %[[B1]] : i1
  //CHECK-NEXT: %[[AxBxC1:.+]] = comb.xor bin %[[AxB1]], %[[C1]] : i1
  //CHECK-NEXT: comb.concat %[[AxBxC1]], %[[AxBxC0]] : i1, i1
  //CHECK-NEXT: comb.concat %[[AB0oAxBC0]], %false : i1, i1
  %0:2 = datapath.compress %a, %b, %c : i2 [3 -> 2]
  hw.output %0#0, %0#1 : i2, i2
}

// CHECK-LABEL: @compressor_add
// TO-ADD-LABEL: @compressor_add
// TO-ADD-NEXT: %c0_i2 = hw.constant 0 : i2
// TO-ADD-NEXT: %[[ADD:.+]] = comb.add bin %a, %b, %c : i2
// TO-ADD-NEXT: hw.output %c0_i2, %[[ADD]] : i2, i2
hw.module @compressor_add(in %a : i2, in %b : i2, in %c : i2, out carry : i2, out save : i2) {
  %0:2 = datapath.compress %a, %b, %c : i2 [3 -> 2]
  hw.output %0#0, %0#1 : i2, i2
}

// CHECK-LABEL: @partial_product
hw.module @partial_product(in %a : i3, in %b : i3, out pp0 : i3, out pp1 : i3, out pp2 : i3) {
  // CHECK-NEXT: %c2_i3 = hw.constant 2 : i3
  // CHECK-NEXT: %c1_i3 = hw.constant 1 : i3
  // CHECK-NEXT: %[[B0:.+]] = comb.extract %b from 0 : (i3) -> i1
  // CHECK-NEXT: %[[B1:.+]] = comb.extract %b from 1 : (i3) -> i1
  // CHECK-NEXT: %[[B2:.+]] = comb.extract %b from 2 : (i3) -> i1
  // CHECK-NEXT: %[[B0R:.+]] = comb.replicate %[[B0]] : (i1) -> i3
  // CHECK-NEXT: %[[PP0:.+]] = comb.and %[[B0R]], %a : i3
  // CHECK-NEXT: %[[B1R:.+]] = comb.replicate %[[B1]] : (i1) -> i3
  // CHECK-NEXT: %[[PP1:.+]]  = comb.and %[[B1R]], %a : i3
  // CHECK-NEXT: comb.shl %[[PP1]], %c1_i3 : i3
  // CHECK-NEXT: %[[B2R:.+]] = comb.replicate %[[B2]] : (i1) -> i3
  // CHECK-NEXT: %[[PP2:.+]] = comb.and %[[B2R]], %a : i3
  // CHECK-NEXT: comb.shl %[[PP2]], %c2_i3 : i3
  %0:3 = datapath.partial_product %a, %b : (i3, i3) -> (i3, i3, i3)
  hw.output %0#0, %0#1, %0#2 : i3, i3, i3
}

// CHECK-LABEL: @partial_product_booth
// FORCE-BOOTH-LABEL: @partial_product_booth
// Constants
// FORCE-BOOTH-NEXT: %true = hw.constant true
// FORCE-BOOTH-NEXT: %false = hw.constant false
// FORCE-BOOTH-NEXT: %c0_i3 = hw.constant 0 : i3
// 2*a
// FORCE-BOOTH-NEXT: %0 = comb.extract %a from 0 : (i3) -> i2
// FORCE-BOOTH-NEXT: %[[TWOA:.+]] = comb.concat %0, %false : i2, i1
// FORCE-BOOTH-NEXT: %[[B0:.+]] = comb.extract %b from 0 : (i3) -> i1
// FORCE-BOOTH-NEXT: %[[B1:.+]] = comb.extract %b from 1 : (i3) -> i1
// FORCE-BOOTH-NEXT: %[[B2:.+]] = comb.extract %b from 2 : (i3) -> i1
// PP0
// FORCE-BOOTH-NEXT: %[[NB0:.+]] = comb.xor bin %[[B0]], %true : i1
// FORCE-BOOTH-NEXT: %[[TWO0:.+]] = comb.and %[[B1]], %[[NB0]] : i1
// FORCE-BOOTH-NEXT: %[[PPOSGN:.+]] = comb.replicate %[[B1]] : (i1) -> i3
// FORCE-BOOTH-NEXT: %[[ONER:.+]] = comb.replicate %[[B0]] : (i1) -> i3
// FORCE-BOOTH-NEXT: %[[TWO0R:.+]] = comb.replicate %[[TWO0]] : (i1) -> i3
// FORCE-BOOTH-NEXT: %[[PP0TWOA:.+]] = comb.and %[[TWO0R]], %[[TWOA]] : i3
// FORCE-BOOTH-NEXT: %[[PP0ONEA:.+]] = comb.and %[[ONER]], %a : i3
// FORCE-BOOTH-NEXT: %[[PP0MAG:.+]] = comb.or bin %[[PP0TWOA]], %[[PP0ONEA]] : i3
// FORCE-BOOTH-NEXT: %[[PP0:.+]] = comb.xor bin %[[PP0MAG]], %[[PPOSGN]] : i3
// PP1
// FORCE-BOOTH-NEXT: %[[B2XORB1:.+]] = comb.xor bin %4, %3 : i1
// FORCE-BOOTH-NEXT: %[[A0:.+]] = comb.extract %a from 0 : (i3) -> i1
// FORCE-BOOTH-NEXT: %[[PP1MSB:.+]] = comb.and %[[B2XORB1]], %[[A0]] : i1
// FORCE-BOOTH-NEXT: %[[PP1:.+]] = comb.concat %[[PP1MSB]], %false, %[[B1]] : i1, i1, i1
// FORCE-BOOTH-NEXT: hw.output %[[PP0]], %[[PP1]], %c0_i3 : i3, i3, i3
hw.module @partial_product_booth(in %a : i3, in %b : i3, out pp0 : i3, out pp1 : i3, out pp2 : i3) {
  %0:3 = datapath.partial_product %a, %b : (i3, i3) -> (i3, i3, i3)
  hw.output %0#0, %0#1, %0#2 : i3, i3, i3
}

// CHECK-LABEL: @partial_product_24
hw.module @partial_product_24(in %a : i24, in %b : i24, out sum : i24) {
  %0:24 = datapath.partial_product %a, %b : (i24, i24) -> (i24, i24, i24, i24, i24, i24, i24, i24, i24, i24, i24, i24, i24, i24, i24, i24, i24, i24, i24, i24, i24, i24, i24, i24)
  %1 = comb.add bin %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7, %0#8, %0#9, %0#10, %0#11, %0#12, %0#13, %0#14, %0#15, %0#16, %0#17, %0#18, %0#19, %0#20, %0#21, %0#22, %0#23 : i24
  hw.output %1 : i24
}

// CHECK-LABEL: @partial_product_25
hw.module @partial_product_25(in %a : i25, in %b : i25, out sum : i25) {
  %0:25 = datapath.partial_product %a, %b : (i25, i25) -> (i25, i25, i25, i25, i25, i25, i25, i25, i25, i25, i25, i25, i25, i25, i25, i25, i25, i25, i25, i25, i25, i25, i25, i25, i25)
  %1 = comb.add bin %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7, %0#8, %0#9, %0#10, %0#11, %0#12, %0#13, %0#14, %0#15, %0#16, %0#17, %0#18, %0#19, %0#20, %0#21, %0#22, %0#23 : i25
  hw.output %1 : i25
}

// CHECK-LABEL: @timing
// TIMING-LABEL: @timing
hw.module @timing(in %a : i1, in %b : i1, in %c : i1, out carry : i1, out save : i1) {
  %and = comb.and bin %a, %b : i1
  // Make sure [[AND]] is pushed to the last stage.
  // CHECK: [[AND:%.+]] = comb.and bin %a, %b : i1
  // CHECK-NEXT: [[XOR1:%.+]] = comb.xor bin [[AND]], %a : i1
  // CHECK-NEXT: [[XOR2:%.+]] = comb.xor bin [[XOR1]], %b : i1
  // CHECK-NEXT: [[XOR3:%.+]] = comb.xor bin [[XOR2]], %c : i1
  // CHECK-NEXT: hw.output [[XOR3]], %false : i1, i1
  // TIMING: [[AND:%.+]] = comb.and bin %a, %b : i1
  // TIMING: [[XOR1:%.+]] = comb.xor bin %a, %b : i1
  // TIMING: [[XOR2:%.+]] = comb.xor bin [[XOR1]], %c : i1
  // TIMING: [[XOR3:%.+]] = comb.xor bin [[XOR2]], [[AND]] : i1
  // TIMING: hw.output [[XOR3]], %false : i1, i1
  %0:2 = datapath.compress %and, %a, %b, %c : i1 [4 -> 2]
  hw.output %0#0, %0#1 : i1, i1
}
