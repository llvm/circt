// RUN: circt-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: @do_nothing
hw.module @do_nothing(in %a : i4, in %b : i4, out carry : i4, out save : i4) {
  // CHECK-NEXT: %[[PP:.+]]:4 = datapath.partial_product %a, %b : (i4, i4) -> (i4, i4, i4, i4)
  // CHECK-NEXT: datapath.compress %[[PP]]#0, %[[PP]]#1, %[[PP]]#2, %[[PP]]#3 : i4 [4 -> 2]
  %0:4 = datapath.partial_product %a, %b : (i4, i4) -> (i4, i4, i4, i4)
  %1:2 = datapath.compress %0#0, %0#1, %0#2, %0#3 : i4 [4 -> 2]
  hw.output %1#0, %1#1 : i4, i4
}

// CHECK-LABEL: @fold_compress
hw.module @fold_compress(in %a : i4, in %b : i4, in %c : i4, in %d : i4, out carry : i4, out save : i4) {
  // CHECK-NEXT: datapath.compress %d, %a, %b, %c : i4 [4 -> 2]
  %0:2 = datapath.compress %a, %b, %c : i4 [3 -> 2]
  %1:2 = datapath.compress %d, %0#0, %0#1 : i4 [3 -> 2]
  hw.output %1#0, %1#1 : i4, i4
}

// CHECK-LABEL: @fold_add
hw.module @fold_add(in %a : i4, in %b : i4, in %c : i4, in %d : i4, out sum : i4) {
  // CHECK-NEXT: %[[COMP:.+]]:2 = datapath.compress %d, %a, %b, %c : i4 [4 -> 2]
  // CHECK-NEXT: comb.add bin %[[COMP]]#0, %[[COMP]]#1 : i4
  %0:2 = datapath.compress %a, %b, %c : i4 [3 -> 2]
  %1 = comb.add %d, %0#0, %0#1 : i4
  hw.output %1 : i4
}

// CHECK-LABEL: @constant_fold_compress
hw.module @constant_fold_compress(in %a : i4, in %b : i4, in %c : i4,
                                  out sum0 : i4, out carry0 : i4, 
                                  out sum1 : i4, out carry1 : i4,  
                                  out sum2 : i4, out carry2 : i4) {
  %c0_i4 = hw.constant 0 : i4
  %0:2 = datapath.compress %a, %b, %c0_i4 : i4 [3 -> 2]
  
  // CHECK-NEXT: %c3_i4 = hw.constant 3 : i4
  // CHECK-NEXT: %[[COMP:.+]]:2 = datapath.compress %a, %b, %c : i4 [3 -> 2]
  %1:2 = datapath.compress %a, %b, %c0_i4, %c : i4 [4 -> 2]

  
  // CHECK-NEXT: %[[COMPFOLD:.+]]:2 = datapath.compress %a, %b, %c, %c3_i4 : i4 [4 -> 2]
  %c1_i4 = hw.constant 1 : i4
  %c2_i4 = hw.constant 2 : i4
  %2:2 = datapath.compress %a, %b, %c1_i4, %c, %c2_i4 : i4 [5 -> 2]
  
  // CHECK-NEXT: hw.output %a, %b, %[[COMP]]#0, %[[COMP]]#1, %[[COMPFOLD]]#0, %[[COMPFOLD]]#1 : i4, i4, i4, i4
  hw.output %0#0, %0#1, %1#0, %1#1, %2#0, %2#1 : i4, i4, i4, i4, i4, i4
}

// CHECK-LABEL: @constant_fold_compress_passthrough
hw.module @constant_fold_compress_passthrough(in %a : i4, in %b : i4, in %c : i4,
                                  out sum0 : i4, out sum1 : i4, out sum2 : i4, 
                                  out sum3 : i4, out sum4 : i4) {
  %c0_i4 = hw.constant 0 : i4
  %0:3 = datapath.compress %a, %b, %c0_i4, %c : i4 [4 -> 3]

  %c1_i4 = hw.constant 1 : i4
  %c2_i4 = hw.constant 2 : i4
  %1:2 = datapath.compress %a, %c1_i4, %c2_i4 : i4 [3 -> 2]
  
  // CHECK-NEXT: %c3_i4 = hw.constant 3 : i4
  // CHECK-NEXT: hw.output %a, %b, %c, %a, %c3_i4 : i4, i4, i4, i4, i4
  hw.output %0#0, %0#1, %0#2, %1#0, %1#1 : i4, i4, i4, i4, i4
}

// CHECK-LABEL: @sext_compress
hw.module @sext_compress(in %a : i8, in %b : i8, in %c : i4, 
                         out sum1 : i8, out carry1 : i8,
                         out sum2 : i8, out carry2 : i8) {
  // CHECK-NEXT: %c-1_i4 = hw.constant -1 : i4
  // CHECK-NEXT: %c-8_i8 = hw.constant -8 : i8
  // CHECK-NEXT: %c0_i4 = hw.constant 0 : i4
  // CHECK-NEXT: %true = hw.constant true
  // CHECK-NEXT: %[[CBASE:.+]] = comb.extract %c from 0 : (i4) -> i3
  // CHECK-NEXT: %[[SIGN:.+]] = comb.extract %c from 3 : (i4) -> i1
  // CHECK-NEXT: %[[NOTSIGN:.+]] = comb.xor bin %[[SIGN]], %true : i1
  // CHECK-NEXT: %[[CEXT:.+]] = comb.concat %c0_i4, %[[NOTSIGN]], %[[CBASE]] : i4, i1, i3
  // CHECK-NEXT: %[[COMP0:.+]]:2 = datapath.compress %a, %b, %[[CEXT]], %c-8_i8 : i8 [4 -> 2]
  %c-1_i8 = hw.constant -1 : i8
  
  // compress(a,b, sext(c))
  %0 = comb.extract %c from 3 : (i4) -> i1
  %1 = comb.replicate %0 : (i1) -> i4
  %2 = comb.concat %1, %c : i4, i4
  %3:2 = datapath.compress %a, %b, %2 : i8 [3 -> 2]
  
  // CHECK-NEXT: %[[NOTC:.+]] = comb.xor bin %c, %c-1_i4 : i4
  // CHECK-NEXT: %[[NOTCBASE:.+]] = comb.extract %[[NOTC]] from 0 : (i4) -> i3
  // CHECK-NEXT: %[[NOTCSIGN:.+]] = comb.extract %[[NOTC]] from 3 : (i4) -> i1
  // CHECK-NEXT: %[[CSIGN:.+]] = comb.xor bin %[[NOTCSIGN]], %true : i1
  // CHECK-NEXT: %[[NOTCEXT:.+]] = comb.concat %c0_i4, %[[CSIGN]], %[[NOTCBASE]] : i4, i1, i3
  // CHECK-NEXT: %[[COMP1:.+]]:2 = datapath.compress %a, %b, %[[NOTCEXT]], %c-8_i8 : i8 [4 -> 2]
  
  // compress(a,b, ~sext(c))
  %4 = comb.xor %2, %c-1_i8 : i8
  %5:2 = datapath.compress %a, %b, %4 : i8 [3 -> 2]
  // CHECK-NEXT: hw.output %[[COMP0]]#0, %[[COMP0]]#1, %[[COMP1]]#0, %[[COMP1]]#1 : i8, i8, i8, i8
  hw.output %3#0, %3#1, %5#0, %5#1 : i8, i8, i8, i8
}

// CHECK-LABEL: @oneext_compress
hw.module @oneext_compress(in %a : i8, in %b : i8, in %c : i4, 
                         out sum1 : i8, out carry1 : i8) {
  
  // CHECK-NEXT: %c-16_i8 = hw.constant -16 : i8
  // CHECK-NEXT: %c0_i4 = hw.constant 0 : i4
  // CHECK-NEXT: %[[CZEXT:.+]] = comb.concat %c0_i4, %c : i4, i4
  // CHECK-NEXT: datapath.compress %a, %b, %[[CZEXT]], %c-16_i8 : i8 [4 -> 2]
  %c-1_i4 = hw.constant -1 : i4
  // compress(a,b, {ones,c})
  %0 = comb.concat %c-1_i4, %c : i4, i4
  %1:2 = datapath.compress %a, %b, %0 : i8 [3 -> 2]
  
  hw.output %1#0, %1#1: i8, i8
}

// CHECK-LABEL: @constant_fold_partial_product
hw.module @constant_fold_partial_product(in %a : i3, in %b : i3, out sum : i4) {
  // CHECK-NEXT: %false = hw.constant false
  // CHECK-NEXT: %[[CONCAT_A:.+]] = comb.concat %false, %a : i1, i3
  // CHECK-NEXT: %[[CONCAT_B:.+]] = comb.concat %false, %b : i1, i3
  // CHECK-NEXT: %[[PP:.+]]:3 = datapath.partial_product %[[CONCAT_A]], %[[CONCAT_B]] : (i4, i4) -> (i4, i4, i4)
  // CHECK-NEXT: %[[COMP:.+]]:2 = datapath.compress %[[PP]]#0, %[[PP]]#1, %[[PP]]#2 : i4 [3 -> 2]
  // CHECK-NEXT: comb.add bin %[[COMP]]#0, %[[COMP]]#1 : i4

  %false = hw.constant false
  %0 = comb.concat %false, %a : i1, i3
  %1 = comb.concat %false, %b : i1, i3
  %2:4 = datapath.partial_product %0, %1 : (i4, i4) -> (i4, i4, i4, i4)
  %3 = comb.add %2#0, %2#1, %2#2, %2#3 : i4
  hw.output %3 : i4
}

// CHECK-LABEL: @partial_product_do_nothing
hw.module @partial_product_do_nothing(in %a : i3, in %b : i4, out sum : i4) {
  // CHECK-NEXT: %false = hw.constant false
  // CHECK-NEXT: %[[CONCAT_A:.+]] = comb.concat %false, %a : i1, i3
  // CHECK-NEXT: %[[PP:.+]]:4 = datapath.partial_product %[[CONCAT_A]], %b : (i4, i4) -> (i4, i4, i4, i4)
  // CHECK-NEXT: %[[COMP:.+]]:2 = datapath.compress %[[PP]]#0, %[[PP]]#1, %[[PP]]#2, %[[PP]]#3 : i4 [4 -> 2]
  // CHECK-NEXT: comb.add bin %[[COMP]]#0, %[[COMP]]#1 : i4
  %false = hw.constant false
  %0 = comb.concat %false, %a : i1, i3
  %1:4 = datapath.partial_product %0, %b : (i4, i4) -> (i4, i4, i4, i4)
  %2:2 = datapath.compress %1#0, %1#1, %1#2, %1#3 : i4 [4 -> 2]
  %3 = comb.add bin %2#0, %2#1 : i4
  hw.output %3 : i4
}

// CHECK-LABEL: @pos_partial_product
hw.module @pos_partial_product(in %a : i4, in %b : i4, in %c : i4, out pp0 : i4, out pp1 : i4, out pp2 : i4, out pp3 : i4) {
  // CHECK-NEXT: datapath.pos_partial_product %a, %b, %c : (i4, i4, i4) -> (i4, i4, i4, i4)
  %0 = comb.add %a, %b : i4
  %1:4 = datapath.partial_product %0, %c : (i4, i4) -> (i4, i4, i4, i4)
  hw.output %1#0, %1#1, %1#2, %1#3 : i4, i4, i4, i4
}

// CHECK-LABEL: @pos_partial_product_reduce
hw.module @pos_partial_product_reduce(in %a : i4, in %b : i3, in %c : i4, out pp0 : i8, out pp1 : i8, out pp2 : i8, out pp3 : i8, out pp4 : i8) {
  // CHECK-NEXT: %c0_i4 = hw.constant 0 : i4
  // CHECK-NEXT: %c0_i5 = hw.constant 0 : i5
  // CHECK-NEXT: %[[AEXT:.+]] = comb.concat %c0_i4, %a : i4, i4
  // CHECK-NEXT: %[[BEXT:.+]] = comb.concat %c0_i5, %b : i5, i3
  // CHECK-NEXT: %[[CEXT:.+]] = comb.concat %c0_i4, %c : i4, i4
  // CHECK-NEXT: datapath.pos_partial_product %[[AEXT]], %[[BEXT]], %[[CEXT]] : (i8, i8, i8) -> (i8, i8, i8, i8, i8)
  %c0_i4 = hw.constant 0 : i4
  %c0_i5 = hw.constant 0 : i5
  %0 = comb.concat %c0_i4, %a : i4, i4
  %1 = comb.concat %c0_i5, %b : i5, i3
  %2 = comb.concat %c0_i4, %c : i4, i4
  %3:8 = datapath.pos_partial_product %0, %1, %2 : (i8, i8, i8) -> (i8, i8, i8, i8, i8, i8, i8, i8)
  hw.output %3#0, %3#1, %3#2, %3#3, %3#4 : i8, i8, i8, i8, i8
}

// CHECK-LABEL: @pos_partial_product_do_nothing
hw.module @pos_partial_product_do_nothing(in %a : i4, in %b : i4, in %c : i4, out pp0 : i4, out pp1 : i4, out pp2 : i4, out pp3 : i4) {
  // CHECK-NEXT: %[[ADD0:.+]] = comb.add %a, %b : i4
  // CHECK-NEXT: %[[ADD1:.+]] = comb.add %a, %c : i4
  // CHECK-NEXT: datapath.partial_product %[[ADD0]], %[[ADD1]] : (i4, i4) -> (i4, i4, i4, i4)
  %0 = comb.add %a, %b : i4
  %1 = comb.add %a, %c : i4
  %2:4 = datapath.partial_product %0, %1 : (i4, i4) -> (i4, i4, i4, i4)
  hw.output %2#0, %2#1, %2#2, %2#3 : i4, i4, i4, i4
}

// CHECK-LABEL: @dont_introduce_compressor
hw.module @dont_introduce_compressor(in %a : i4, in %b : i4, in %c: i4, out sum : i4) {
  // CHECK-NOT: datapath.compress
  // CHECK-NEXT: comb.add
  // CHECK-NEXT: hw.output
  %0:4 = datapath.partial_product %a, %b : (i4, i4) -> (i4, i4, i4, i4)
  %1 = comb.add %a, %b, %c : i4
  hw.output %1 : i4
}
