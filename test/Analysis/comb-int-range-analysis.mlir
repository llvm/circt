// RUN: circt-opt %s --test-comb-int-range-analysis | FileCheck %s

// CHECK-LABEL: @basic_csa
hw.module @basic_csa(in %a : i1, in %b : i1, in %c : i1, out add_abc : i3) {
  // CHECK-NEXT: %c0_i2 = hw.constant 0 : i2 {smax = 0 : i2, smin = 0 : i2, umax = 0 : ui2, umin = 0 : ui2}
  // CHECK-NEXT: %false = hw.constant false 
  // CHECK-NEXT: %[[A_EXT:.+]] = comb.concat %false, %a {smax = 1 : i2, smin = 0 : i2, umax = 1 : ui2, umin = 0 : ui2} : i1, i1
  // CHECK-NEXT: %[[B_EXT:.+]] = comb.concat %false, %b {smax = 1 : i2, smin = 0 : i2, umax = 1 : ui2, umin = 0 : ui2} : i1, i1
  // CHECK-NEXT: %[[ADD:.+]] = comb.add %[[A_EXT]], %[[B_EXT]] {smax = 1 : i2, smin = -2 : i2, umax = 2 : ui2, umin = 0 : ui2} : i2
  // CHECK-NEXT: %[[ADD_EXT:.+]] = comb.concat %false, %[[ADD]] {smax = 2 : i3, smin = 0 : i3, umax = 2 : ui3, umin = 0 : ui3}  : i1, i2
  // CHECK-NEXT: %[[C_EXT:.+]] = comb.concat %c0_i2, %c {smax = 1 : i3, smin = 0 : i3, umax = 1 : ui3, umin = 0 : ui3} : i2, i1
  // CHECK-NEXT: %[[ADD1:.+]] = comb.add %[[ADD_EXT]], %[[C_EXT]] {smax = 3 : i3, smin = 0 : i3, umax = 3 : ui3, umin = 0 : ui3} : i3
  %c0_i2 = hw.constant 0 : i2
  %false = hw.constant false
  %0 = comb.concat %false, %a : i1, i1
  %1 = comb.concat %false, %b : i1, i1
  %2 = comb.add %0, %1 : i2
  %3 = comb.concat %false, %2 : i1, i2
  %4 = comb.concat %c0_i2, %c : i2, i1
  %5 = comb.add %3, %4 : i3
  hw.output %5 : i3
}

// CHECK-LABEL: @basic_mux
hw.module @basic_mux(in %a : i3, in %b : i3, in %sel : i1, out y : i4) {
  // CHECK-NEXT: %false = hw.constant false {smax = false, smin = false, umax = 0 : ui1, umin = 0 : ui1}
  // CHECK-NEXT: %true = hw.constant true {smax = true, smin = true, umax = 1 : ui1, umin = 1 : ui1}
  // CHECK-NEXT: %[[A_EXT:.+]] = comb.concat %true, %a {smax = -1 : i4, smin = -8 : i4, umax = 15 : ui4, umin = 8 : ui4} : i1, i3
  // CHECK-NEXT: %[[B_EXT:.+]] = comb.concat %false, %b {smax = 7 : i4, smin = 0 : i4, umax = 7 : ui4, umin = 0 : ui4} : i1, i3
  // CHECK-NEXT: %[[MUX:.+]] = comb.mux %sel, %[[A_EXT]], %[[B_EXT]] {smax = 7 : i4, smin = -8 : i4, umax = 15 : ui4, umin = 0 : ui4} : i4
  %false = hw.constant false
  %true = hw.constant true
  %0 = comb.concat %true, %a : i1, i3
  %1 = comb.concat %false, %b : i1, i3
  %2 = comb.mux %sel, %0, %1 : i4
  hw.output %2 : i4
}

// CHECK-LABEL: @basic_fma
hw.module @basic_fma(in %a : i4, in %b : i4, in %c : i4, out d : i9) {
  // CHECK-NEXT: %c0_i5 = hw.constant 0 : i5 {smax = 0 : i5, smin = 0 : i5, umax = 0 : ui5, umin = 0 : ui5}
  // CHECK-NEXT: %[[A_EXT:.+]] = comb.concat %c0_i5, %a {smax = 15 : i9, smin = 0 : i9, umax = 15 : ui9, umin = 0 : ui9} : i5, i4
  // CHECK-NEXT: %[[B_EXT:.+]] = comb.concat %c0_i5, %b {smax = 15 : i9, smin = 0 : i9, umax = 15 : ui9, umin = 0 : ui9} : i5, i4
  // CHECK-NEXT: %[[MUL:.+]] = comb.mul %[[A_EXT]], %[[B_EXT]] {smax = 225 : i9, smin = 0 : i9, umax = 225 : ui9, umin = 0 : ui9} : i9
  // CHECK-NEXT: %[[C_EXT:.+]] = comb.concat %c0_i5, %c {smax = 15 : i9, smin = 0 : i9, umax = 15 : ui9, umin = 0 : ui9} : i5, i4
  // CHECK-NEXT: %[[ADD:.+]] = comb.add %[[MUL]], %[[C_EXT]] {smax = 240 : i9, smin = 0 : i9, umax = 240 : ui9, umin = 0 : ui9} : i9
  %c0_i5 = hw.constant 0 : i5
  %0 = comb.concat %c0_i5, %a : i5, i4
  %1 = comb.concat %c0_i5, %b : i5, i4
  %2 = comb.mul %0, %1 : i9
  %3 = comb.concat %c0_i5, %c : i5, i4
  %4 = comb.add %2, %3 : i9
  hw.output %4 : i9
}

// CHECK-LABEL: @const_sub
hw.module @const_sub(in %a : i8, out sub_res : i10) {
  // CHECK-NEXT: %c256_i10 = hw.constant 256 : i10 {smax = 256 : i10, smin = 256 : i10, umax = 256 : ui10, umin = 256 : ui10}
  // CHECK-NEXT: %c0_i2 = hw.constant 0 : i2 {smax = 0 : i2, smin = 0 : i2, umax = 0 : ui2, umin = 0 : ui2}
  // CHECK-NEXT: %[[A_EXT:.+]] = comb.concat %c0_i2, %a {smax = 255 : i10, smin = 0 : i10, umax = 255 : ui10, umin = 0 : ui10} : i2, i8
  // CHECK-NEXT: %[[SUB:.+]] = comb.sub %c256_i10, %[[A_EXT]] {smax = 256 : i10, smin = 1 : i10, umax = 256 : ui10, umin = 1 : ui10} : i10
  %c256_i10 = hw.constant 256 : i10
  %c0_i2 = hw.constant 0 : i2
  %0 = comb.concat %c0_i2, %a : i2, i8
  %1 = comb.sub %c256_i10, %0 : i10
  hw.output %1 : i10
}

// CHECK-LABEL: @logical_ops
hw.module @logical_ops(in %a : i8, in %b : i9, in %c : i10, in %d : i16, out res : i18) {
  // CHECK-NEXT: %c0_i2 = hw.constant 0 : i2 {smax = 0 : i2, smin = 0 : i2, umax = 0 : ui2, umin = 0 : ui2}
  // CHECK-NEXT: %false = hw.constant false {smax = false, smin = false, umax = 0 : ui1, umin = 0 : ui1}
  // CHECK-NEXT: %c0_i9 = hw.constant 0 : i9 {smax = 0 : i9, smin = 0 : i9, umax = 0 : ui9, umin = 0 : ui9}
  // CHECK-NEXT: %c0_i8 = hw.constant 0 : i8 {smax = 0 : i8, smin = 0 : i8, umax = 0 : ui8, umin = 0 : ui8}
  // CHECK-NEXT: %[[A_EXT:.+]] = comb.concat %c0_i9, %a {smax = 255 : i17, smin = 0 : i17, umax = 255 : ui17, umin = 0 : ui17} : i9, i8
  // CHECK-NEXT: %[[B_EXT:.+]] = comb.concat %c0_i8, %b {smax = 511 : i17, smin = 0 : i17, umax = 511 : ui17, umin = 0 : ui17} : i8, i9
  // CHECK-NEXT: %[[AND:.+]] = comb.and %[[A_EXT]], %[[B_EXT]] {smax = 255 : i17, smin = 0 : i17, umax = 255 : ui17, umin = 0 : ui17} : i17
  // CHECK-NEXT: %[[AND_EXT:.+]] = comb.concat %false, %[[AND]] {smax = 255 : i18, smin = 0 : i18, umax = 255 : ui18, umin = 0 : ui18} : i1, i17
  // CHECK-NEXT: %[[C_EXT:.+]] = comb.concat %c0_i8, %c {smax = 1023 : i18, smin = 0 : i18, umax = 1023 : ui18, umin = 0 : ui18} : i8, i10
  // CHECK-NEXT: %[[OR:.+]] = comb.or %[[AND_EXT]], %[[C_EXT]] {smax = 1023 : i18, smin = 0 : i18, umax = 1023 : ui18, umin = 0 : ui18} : i18
  // CHECK-NEXT: %[[D_EXT:.+]] = comb.concat %c0_i2, %d {smax = 65535 : i18, smin = 0 : i18, umax = 65535 : ui18, umin = 0 : ui18} : i2, i16
  // CHECK-NEXT: %[[ADD:.+]] = comb.add %[[OR]], %[[D_EXT]] {smax = 66558 : i18, smin = 0 : i18, umax = 66558 : ui18, umin = 0 : ui18} : i18
  %c0_i2 = hw.constant 0 : i2
  %false = hw.constant false
  %c0_i9 = hw.constant 0 : i9
  %c0_i8 = hw.constant 0 : i8
  %0 = comb.concat %c0_i9, %a : i9, i8
  %1 = comb.concat %c0_i8, %b : i8, i9
  %2 = comb.and %0, %1 : i17
  %3 = comb.concat %false, %2 : i1, i17
  %4 = comb.concat %c0_i8, %c : i8, i10
  %5 = comb.or %3, %4 : i18
  %6 = comb.concat %c0_i2, %d : i2, i16
  %7 = comb.add %5, %6 : i18
  hw.output %7 : i18
}

// CHECK-LABEL: @variadic_ops
hw.module @variadic_ops(in %a : i2, in %b : i2, in %c : i2) {
  // CHECK-NEXT: %c0_i2 = hw.constant 0 : i2 {smax = 0 : i2, smin = 0 : i2, umax = 0 : ui2, umin = 0 : ui2}
  // CHECK-NEXT: %[[A_EXT2:.+]] = comb.concat %c0_i2, %a {smax = 3 : i4, smin = 0 : i4, umax = 3 : ui4, umin = 0 : ui4} : i2, i2
  // CHECK-NEXT: %[[B_EXT2:.+]] = comb.concat %c0_i2, %b {smax = 3 : i4, smin = 0 : i4, umax = 3 : ui4, umin = 0 : ui4} : i2, i2
  // CHECK-NEXT: %[[C_EXT2:.+]] = comb.concat %c0_i2, %c {smax = 3 : i4, smin = 0 : i4, umax = 3 : ui4, umin = 0 : ui4} : i2, i2
  // CHECK-NEXT: %[[ADD:.+]] = comb.add %[[A_EXT2]], %[[B_EXT2]], %[[C_EXT2]] {smax = 7 : i4, smin = -8 : i4, umax = 9 : ui4, umin = 0 : ui4} : i4
  // CHECK-NEXT: %c0_i3 = hw.constant 0 : i3 {smax = 0 : i3, smin = 0 : i3, umax = 0 : ui3, umin = 0 : ui3}
  // CHECK-NEXT: %[[A_EXT3:.+]] = comb.concat %c0_i3, %a {smax = 3 : i5, smin = 0 : i5, umax = 3 : ui5, umin = 0 : ui5} : i3, i2
  // CHECK-NEXT: %[[B_EXT3:.+]] = comb.concat %c0_i3, %b {smax = 3 : i5, smin = 0 : i5, umax = 3 : ui5, umin = 0 : ui5} : i3, i2
  // CHECK-NEXT: %[[C_EXT3:.+]] = comb.concat %c0_i3, %c {smax = 3 : i5, smin = 0 : i5, umax = 3 : ui5, umin = 0 : ui5} : i3, i2
  // CHECK-NEXT: %[[MUL:.+]] = comb.mul %[[A_EXT3]], %[[B_EXT3]], %[[C_EXT3]] {smax = 15 : i5, smin = -16 : i5, umax = 27 : ui5, umin = 0 : ui5} : i5
  // CHECK-NEXT: %[[AND:.+]] = comb.and %a, %b, %c {smax = 1 : i2, smin = -2 : i2, umax = 3 : ui2, umin = 0 : ui2} : i2
  // CHECK-NEXT: %[[OR:.+]] = comb.or %a, %b, %c {smax = 1 : i2, smin = -2 : i2, umax = 3 : ui2, umin = 0 : ui2} : i2
  // CHECK-NEXT: %[[XOR:.+]] = comb.xor %a, %b, %c {smax = 1 : i2, smin = -2 : i2, umax = 3 : ui2, umin = 0 : ui2} : i2
  // CHECK-NEXT: hw.output
  %c0_i2 = hw.constant 0 : i2
  %0 = comb.concat %c0_i2, %a : i2, i2
  %1 = comb.concat %c0_i2, %b : i2, i2
  %2 = comb.concat %c0_i2, %c : i2, i2
  %3 = comb.add %0, %1, %2 : i4
  %c0_i3 = hw.constant 0 : i3
  %4 = comb.concat %c0_i3, %a : i3, i2
  %5 = comb.concat %c0_i3, %b : i3, i2
  %6 = comb.concat %c0_i3, %c : i3, i2
  %7 = comb.mul %4, %5, %6 : i5
  %8  = comb.and %a, %b, %c : i2
  %9 = comb.or  %a, %b, %c : i2
  %10 = comb.xor %a, %b, %c : i2
  hw.output
}

// CHECK-LABEL: @replicate_extract
hw.module @replicate_extract(in %a : i3, in %b : i3, in %sel : i1) {
  // CHECK-NEXT: %c0_i2 = hw.constant 0 : i2 {smax = 0 : i2, smin = 0 : i2, umax = 0 : ui2, umin = 0 : ui2}
  // CHECK-NEXT: %[[EXT_A:.+]] = comb.extract %a from 1 {smax = 1 : i2, smin = -2 : i2, umax = 3 : ui2, umin = 0 : ui2} : (i3) -> i2
  // CHECK-NEXT: %[[REPL_A:.+]] = comb.replicate %[[EXT_A]] {smax = 7 : i4, smin = -8 : i4, umax = 15 : ui4, umin = 0 : ui4} : (i2) -> i4
  // CHECK-NEXT: %[[REPL_SEL:.+]] = comb.replicate %sel {smax = 1 : i2, smin = -2 : i2, umax = 3 : ui2, umin = 0 : ui2} : (i1) -> i2
  // CHECK-NEXT: %[[EXT_OUT:.+]] = comb.extract %[[REPL_A]] from 1 {smax = 1 : i2, smin = -2 : i2, umax = 3 : ui2, umin = 0 : ui2} : (i4) -> i2
  %c0_i2 = hw.constant 0 : i2
  %0 = comb.extract %a from 1 : (i3) -> i2
  %1 = comb.replicate %0 : (i2) -> i4
  %2 = comb.replicate %sel : (i1) -> i2
  %3 = comb.extract %1 from 1 : (i4) -> i2
  hw.output
}

// CHECK-LABEL: @comp_predicates
hw.module @comp_predicates(in %a : i3, in %b : i3) {
  // CHECK-NEXT: %c0_i2 = hw.constant 0 : i2 {smax = 0 : i2, smin = 0 : i2, umax = 0 : ui2, umin = 0 : ui2}
  // CHECK-NEXT: %c0_i3 = hw.constant 0 : i3 {smax = 0 : i3, smin = 0 : i3, umax = 0 : ui3, umin = 0 : ui3}
  // CHECK-NEXT: %c-1_i3 = hw.constant -1 : i3 {smax = -1 : i3, smin = -1 : i3, umax = 7 : ui3, umin = 7 : ui3}
  // CHECK-NEXT: %[[ULT:.+]] = comb.icmp ult %a, %c-1_i3 {smax = false, smin = true, umax = 1 : ui1, umin = 0 : ui1} : i3
  // CHECK-NEXT: %[[ULE:.+]] = comb.icmp ule %a, %c-1_i3 {smax = true, smin = true, umax = 1 : ui1, umin = 1 : ui1} : i3
  // CHECK-NEXT: %[[UGT:.+]] = comb.icmp ugt %a, %b {smax = false, smin = true, umax = 1 : ui1, umin = 0 : ui1} : i3
  // CHECK-NEXT: %[[UGE:.+]] = comb.icmp uge %a, %c0_i3 {smax = true, smin = true, umax = 1 : ui1, umin = 1 : ui1} : i3
  // CHECK-NEXT: %[[SLT:.+]] = comb.icmp slt %a, %b {smax = false, smin = true, umax = 1 : ui1, umin = 0 : ui1} : i3
  // CHECK-NEXT: %[[SLE:.+]] = comb.icmp sle %a, %b {smax = false, smin = true, umax = 1 : ui1, umin = 0 : ui1} : i3
  // CHECK-NEXT: %[[SGT:.+]] = comb.icmp sgt %a, %b {smax = false, smin = true, umax = 1 : ui1, umin = 0 : ui1} : i3
  // CHECK-NEXT: %[[SGE:.+]] = comb.icmp sge %a, %c0_i3 {smax = false, smin = true, umax = 1 : ui1, umin = 0 : ui1} : i3
  // CHECK-NEXT: %[[EQ:.+]] = comb.icmp eq %a, %b {smax = false, smin = true, umax = 1 : ui1, umin = 0 : ui1} : i3
  // CHECK-NEXT: %[[NE:.+]] = comb.icmp ne %a, %b {smax = false, smin = true, umax = 1 : ui1, umin = 0 : ui1} : i3
  %c0_i2 = hw.constant 0 : i2
  %c0_i3 = hw.constant 0 : i3
  %c7_i3 = hw.constant 7 : i3
  %0 = comb.icmp ult %a, %c7_i3 : i3
  %1 = comb.icmp ule %a, %c7_i3  : i3
  %2 = comb.icmp ugt %a, %b : i3
  %3 = comb.icmp uge %a, %c0_i3 : i3
  %4 = comb.icmp slt %a, %b : i3
  %5 = comb.icmp sle %a, %b  : i3
  %6 = comb.icmp sgt %a, %b : i3
  %7 = comb.icmp sge %a, %c0_i3 : i3
  %8 = comb.icmp eq %a, %b  : i3
  %9 = comb.icmp ne %a, %b  : i3
  hw.output
}
