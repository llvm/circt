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

// CHECK-LABEL: @extract_ranges
hw.module @extract_ranges(in %a : i8, in %b : i8, in %sel : i1) {
  // CHECK-NEXT: %c0_i8 = hw.constant 0 : i8 {smax = 0 : i8, smin = 0 : i8, umax = 0 : ui8, umin = 0 : ui8}
  // CHECK-NEXT: %[[A16:.+]] = comb.concat %c0_i8, %a {smax = 255 : i16, smin = 0 : i16, umax = 255 : ui16, umin = 0 : ui16} : i8, i8
  // CHECK-NEXT: %[[B16:.+]] = comb.concat %c0_i8, %b {smax = 255 : i16, smin = 0 : i16, umax = 255 : ui16, umin = 0 : ui16} : i8, i8
  // CHECK-NEXT: %[[PROD:.+]] = comb.mul %[[A16]], %[[B16]] {smax = 32767 : i16, smin = -32768 : i16, umax = 65025 : ui16, umin = 0 : ui16} : i16
  // The dropped bit 15 differs between umin (0) and umax (0xFE01), so bits [14:8] span the full i7 range.
  // CHECK-NEXT: %{{.+}} = comb.extract %[[PROD]] from 8 {smax = 63 : i7, smin = -64 : i7, umax = 127 : ui7, umin = 0 : ui7} : (i16) -> i7
  // The slice stays exact when the dropped high bits match (0x0102..0x0105 yields [2, 5]).
  // CHECK-NEXT: %c258_i16 = hw.constant 258 : i16
  // CHECK-NEXT: %c261_i16 = hw.constant 261 : i16
  // CHECK-NEXT: %[[SAME_HI:.+]] = comb.mux %sel, %c258_i16, %c261_i16
  // CHECK-NEXT: %{{.+}} = comb.extract %[[SAME_HI]] from 0 {smax = 5 : i8, smin = 2 : i8, umax = 5 : ui8, umin = 2 : ui8} : (i16) -> i8
  // The signed range [-4, 3] crosses zero (unsigned [0, 255]), so arithmetic right-shifting and truncating preserves the signed bounds.
  // CHECK-NEXT: %c-4_i8 = hw.constant -4 : i8
  // CHECK-NEXT: %c3_i8 = hw.constant 3 : i8
  // CHECK-NEXT: %[[SIGNED:.+]] = comb.mux %sel, %c-4_i8, %c3_i8
  // CHECK-NEXT: %{{.+}} = comb.extract %[[SIGNED]] from 0 {smax = 3 : i4, smin = -4 : i4, umax = 15 : ui4, umin = 0 : ui4} : (i8) -> i4
  // CHECK-NEXT: %{{.+}} = comb.extract %[[SIGNED]] from 2 {smax = 0 : i6, smin = -1 : i6, umax = 63 : ui6, umin = 0 : ui6} : (i8) -> i6
  // Zero-width and full-width extracts return a constant zero range and the original input range, respectively.
  // CHECK-NEXT: %{{.+}} = comb.extract %a from 8 {smax = 0 : i0, smin = 0 : i0, umax = 0 : ui0, umin = 0 : ui0} : (i8) -> i0
  // CHECK-NEXT: %{{.+}} = comb.extract %[[SAME_HI]] from 0 {smax = 261 : i16, smin = 258 : i16, umax = 261 : ui16, umin = 258 : ui16} : (i16) -> i16
  %c0_i8 = hw.constant 0 : i8
  %a16 = comb.concat %c0_i8, %a : i8, i8
  %b16 = comb.concat %c0_i8, %b : i8, i8
  %prod = comb.mul %a16, %b16 : i16
  %m_pre = comb.extract %prod from 8 : (i16) -> i7

  %c258_i16 = hw.constant 258 : i16
  %c261_i16 = hw.constant 261 : i16
  %same_hi = comb.mux %sel, %c258_i16, %c261_i16 : i16
  %exact_lo = comb.extract %same_hi from 0 : (i16) -> i8

  %cm4_i8 = hw.constant -4 : i8
  %c3_i8 = hw.constant 3 : i8
  %signed_range = comb.mux %sel, %cm4_i8, %c3_i8 : i8
  %signed_lo = comb.extract %signed_range from 0 : (i8) -> i4
  %signed_hi = comb.extract %signed_range from 2 : (i8) -> i6

  %zero_w = comb.extract %a from 8 : (i8) -> i0
  %full_w = comb.extract %same_hi from 0 : (i16) -> i16
  hw.output
}

// Test interior slices where both lowBit > 0 and lowBit + resWidth < srcWidth.
// CHECK-LABEL: @extract_middle_ranges
hw.module @extract_middle_ranges(in %sel : i1) {
  // CHECK-NEXT: %c4656_i16 = hw.constant 4656 : i16
  // CHECK-NEXT: %c4703_i16 = hw.constant 4703 : i16
  // CHECK-NEXT: %[[MID_SAME_HI:.+]] = comb.mux %sel, %c4656_i16, %c4703_i16
  // The dropped high bits [15:8] match (0x12), so bits [7:4] of 0x1230..0x125F stay exact at [3, 5].
  // CHECK-NEXT: %{{.+}} = comb.extract %[[MID_SAME_HI]] from 4 {smax = 5 : i4, smin = 3 : i4, umax = 5 : ui4, umin = 3 : ui4} : (i16) -> i4
  // Extracting a single interior bit exercises resWidth == 1 (bit 9 of 0x1230..0x125F is always 1).
  // CHECK-NEXT: %{{.+}} = comb.extract %[[MID_SAME_HI]] from 9 {smax = true, smin = true, umax = 1 : ui1, umin = 1 : ui1} : (i16) -> i1
  // The dropped high bits [15:8] match (0x12), while bits [7:4] of 0x1260..0x129F cross the i4 sign boundary (7 -> -8).
  // CHECK-NEXT: %c4704_i16 = hw.constant 4704 : i16
  // CHECK-NEXT: %c4767_i16 = hw.constant 4767 : i16
  // CHECK-NEXT: %[[MID_SIGN_CROSS:.+]] = comb.mux %sel, %c4704_i16, %c4767_i16
  // CHECK-NEXT: %{{.+}} = comb.extract %[[MID_SIGN_CROSS]] from 4 {smax = 7 : i4, smin = -8 : i4, umax = 9 : ui4, umin = 6 : ui4} : (i16) -> i4
  // The dropped bit 8 differs between 0x1230 and 0x135F, so bits [7:4] roll over and widen to the full i4 range.
  // CHECK-NEXT: %c4959_i16 = hw.constant 4959 : i16
  // CHECK-NEXT: %[[MID_DIFF_HI:.+]] = comb.mux %sel, %c4656_i16, %c4959_i16
  // CHECK-NEXT: %{{.+}} = comb.extract %[[MID_DIFF_HI]] from 4 {smax = 7 : i4, smin = -8 : i4, umax = 15 : ui4, umin = 0 : ui4} : (i16) -> i4
  // The signed range [-16, 15] crosses zero; shifting right by 2 yields [-4, 3], whose dropped high bits are sign-extension bits.
  // CHECK-NEXT: %c-16_i8 = hw.constant -16 : i8
  // CHECK-NEXT: %c15_i8 = hw.constant 15 : i8
  // CHECK-NEXT: %[[MID_SIGNED:.+]] = comb.mux %sel, %c-16_i8, %c15_i8
  // CHECK-NEXT: %{{.+}} = comb.extract %[[MID_SIGNED]] from 2 {smax = 3 : i4, smin = -4 : i4, umax = 15 : ui4, umin = 0 : ui4} : (i8) -> i4
  // Shifting [-64, 12] right by 2 yields [-16, 3], which overflows i4 and must widen rather than truncating to [0, 3].
  // CHECK-NEXT: %c-64_i8 = hw.constant -64 : i8
  // CHECK-NEXT: %c12_i8 = hw.constant 12 : i8
  // CHECK-NEXT: %[[MID_SIGNED_ROLL:.+]] = comb.mux %sel, %c-64_i8, %c12_i8
  // CHECK-NEXT: %{{.+}} = comb.extract %[[MID_SIGNED_ROLL]] from 2 {smax = 7 : i4, smin = -8 : i4, umax = 15 : ui4, umin = 0 : ui4} : (i8) -> i4
  // For the negative range [-16, -4], lshr(2) gives [60, 63] while ashr(2) gives [-4, -1]; truncating each before intersecting preserves the exact bounds.
  // CHECK-NEXT: %c-4_i8 = hw.constant -4 : i8
  // CHECK-NEXT: %[[MID_NEG:.+]] = comb.mux %sel, %c-16_i8, %c-4_i8
  // CHECK-NEXT: %{{.+}} = comb.extract %[[MID_NEG]] from 2 {smax = -1 : i4, smin = -4 : i4, umax = 15 : ui4, umin = 12 : ui4} : (i8) -> i4
  %c4656_i16 = hw.constant 4656 : i16
  %c4703_i16 = hw.constant 4703 : i16
  %mid_same_hi = comb.mux %sel, %c4656_i16, %c4703_i16 : i16
  %mid_exact = comb.extract %mid_same_hi from 4 : (i16) -> i4
  %mid_bit = comb.extract %mid_same_hi from 9 : (i16) -> i1

  %c4704_i16 = hw.constant 4704 : i16
  %c4767_i16 = hw.constant 4767 : i16
  %mid_sign_cross = comb.mux %sel, %c4704_i16, %c4767_i16 : i16
  %mid_sign_cross_ext = comb.extract %mid_sign_cross from 4 : (i16) -> i4

  %c4959_i16 = hw.constant 4959 : i16
  %mid_diff_hi = comb.mux %sel, %c4656_i16, %c4959_i16 : i16
  %mid_rollover = comb.extract %mid_diff_hi from 4 : (i16) -> i4

  %cm16_i8 = hw.constant -16 : i8
  %c15_i8 = hw.constant 15 : i8
  %mid_signed = comb.mux %sel, %cm16_i8, %c15_i8 : i8
  %mid_signed_ext = comb.extract %mid_signed from 2 : (i8) -> i4

  %cm64_i8 = hw.constant -64 : i8
  %c12_i8 = hw.constant 12 : i8
  %mid_signed_roll = comb.mux %sel, %cm64_i8, %c12_i8 : i8
  %mid_signed_roll_ext = comb.extract %mid_signed_roll from 2 : (i8) -> i4

  %cm4_i8 = hw.constant -4 : i8
  %mid_neg = comb.mux %sel, %cm16_i8, %cm4_i8 : i8
  %mid_neg_ext = comb.extract %mid_neg from 2 : (i8) -> i4
  hw.output
}
