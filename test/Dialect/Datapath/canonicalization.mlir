// RUN: circt-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: @do_nothing
hw.module @do_nothing(in %a : i4, in %b : i4, out carry : i4, out save : i4) {
  // CHECK-NEXT: %[[PP:.+]]:4 = datapath.pp %a, %b : 4 x i4
  // CHECK-NEXT: datapath.compress %[[PP]]#0, %[[PP]]#1, %[[PP]]#2, %[[PP]]#3 : 4 x i4 -> (i4, i4)
  %0:4 = datapath.pp %a, %b : 4 x i4
  %1:2 = datapath.compress %0#0, %0#1, %0#2, %0#3 : 4 x i4 -> (i4, i4)
  hw.output %1#0, %1#1 : i4, i4
}

// CHECK-LABEL: @fold_compress
hw.module @fold_compress(in %a : i4, in %b : i4, in %c : i4, in %d : i4, out carry : i4, out save : i4) {
  // CHECK-NEXT: datapath.compress %d, %a, %b, %c : 4 x i4 -> (i4, i4)
  %0:2 = datapath.compress %a, %b, %c : 3 x i4 -> (i4, i4)
  %1:2 = datapath.compress %d, %0#0, %0#1 : 3 x i4 -> (i4, i4)
  hw.output %1#0, %1#1 : i4, i4
}

// CHECK-LABEL: @fold_add
hw.module @fold_add(in %a : i4, in %b : i4, in %c : i4, in %d : i4, out sum : i4) {
  // CHECK-NEXT: %[[COMP:.+]]:2 = datapath.compress %d, %a, %b, %c : 4 x i4 -> (i4, i4)
  // CHECK-NEXT: comb.add bin %[[COMP]]#0, %[[COMP]]#1 : i4
  %0:2 = datapath.compress %a, %b, %c : 3 x i4 -> (i4, i4)
  %1 = comb.add %d, %0#0, %0#1 : i4
  hw.output %1 : i4
}

// CHECK-LABEL: @constant_fold_compress
hw.module @constant_fold_compress(in %a : i4, in %b : i4, in %c : i4,
                                  out sum0 : i4, out carry0 : i4, out sum1 : i4, out carry1 : i4) {
  // CHECK-NEXT: %c0_i4 = hw.constant 0 : i4 
  // CHECK-NEXT: %[[ADD:.+]] = comb.add bin %a, %b : i4
  %c0_i4 = hw.constant 0 : i4
  %0:2 = datapath.compress %a, %b, %c0_i4 : 3 x i4 -> (i4, i4)
  
  // CHECK-NEXT: %[[COMP:.+]]:2 = datapath.compress %a, %b, %c : 3 x i4 -> (i4, i4)
  %1:2 = datapath.compress %a, %b, %c0_i4, %c : 4 x i4 -> (i4, i4)
  
  // CHECK-NEXT: hw.output %[[ADD]], %c0_i4, %[[COMP]]#0, %[[COMP]]#1 : i4, i4, i4, i4
  hw.output %0#0, %0#1, %1#0, %1#1 : i4, i4, i4, i4
}

// CHECK-LABEL: @constant_fold_compress_unsupported
hw.module @constant_fold_compress_unsupported(in %a : i4, in %b : i4, in %c : i4,
                                  out sum0 : i4, out sum1 : i4, out sum2 : i4) {
  // CHECK-NEXT: %c0_i4 = hw.constant 0 : i4 
  // CHECK-NEXT: datapath.compress %a, %b, %c, %c0_i4 : 4 x i4 -> (i4, i4, i4)
  %c0_i4 = hw.constant 0 : i4
  %0:3 = datapath.compress %a, %b, %c0_i4, %c : 4 x i4 -> (i4, i4, i4)
  hw.output %0#0, %0#1, %0#2 : i4, i4, i4
}

// CHECK-LABEL: @constant_fold_partial_product
hw.module @constant_fold_partial_product(in %a : i3, in %b : i3, out sum : i4) {
  // CHECK-NEXT: %false = hw.constant false
  // CHECK-NEXT: %[[CONCAT_A:.+]] = comb.concat %false, %a : i1, i3
  // CHECK-NEXT: %[[CONCAT_B:.+]] = comb.concat %false, %b : i1, i3
  // CHECK-NEXT: %[[PP:.+]]:3 = datapath.pp %[[CONCAT_A]], %[[CONCAT_B]] : 3 x i4
  // CHECK-NEXT: comb.add %[[PP]]#0, %[[PP]]#1, %[[PP]]#2 : i4
  %false = hw.constant false
  %0 = comb.concat %false, %a : i1, i3
  %1 = comb.concat %false, %b : i1, i3
  %2:4 = datapath.pp %0, %1 : 4 x i4
  %3 = comb.add %2#0, %2#1, %2#2, %2#3 : i4
  hw.output %3 : i4
}

// CHECK-LABEL: @partial_product_do_nothing
hw.module @partial_product_do_nothing(in %a : i3, in %b : i4, out sum : i4) {
  // CHECK-NEXT: %false = hw.constant false
  // CHECK-NEXT: %[[CONCAT_A:.+]] = comb.concat %false, %a : i1, i3
  // CHECK-NEXT: %[[PP:.+]]:4 = datapath.pp %[[CONCAT_A]], %b : 4 x i4
  // CHECK-NEXT: comb.add %[[PP]]#0, %[[PP]]#1, %[[PP]]#2, %[[PP]]#3 : i4
  %false = hw.constant false
  %0 = comb.concat %false, %a : i1, i3
  %1:4 = datapath.pp %0, %b : 4 x i4
  %2 = comb.add %1#0, %1#1, %1#2, %1#3 : i4
  hw.output %2 : i4
}
