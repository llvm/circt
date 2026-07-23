// REQUIRES: z3

// RUN: circt-opt %s --convert-datapath-to-comb -o %t.mlir
// RUN: circt-lec.sh %t.mlir %s -c1=partial_product_5 -c2=partial_product_5
hw.module @partial_product_5(in %a : i5, in %b : i5, out sum : i5) {
  %0:5 = datapath.partial_product %a, %b : (i5, i5) -> (i5, i5, i5, i5, i5)
  %1 = comb.add bin %0#0, %0#1, %0#2, %0#3, %0#4 : i5
  hw.output %1 : i5
}

// RUN: circt-lec.sh %t.mlir %s -c1=partial_product_4 -c2=partial_product_4
hw.module @partial_product_4(in %a : i4, in %b : i4, out sum : i4) {
  %0:4 = datapath.partial_product %a, %b : (i4, i4) -> (i4, i4, i4, i4)
  %1 = comb.add bin %0#0, %0#1, %0#2, %0#3 : i4
  hw.output %1 : i4
}

// RUN: circt-lec.sh %t.mlir %s -c1=partial_product_zext -c2=partial_product_zext
hw.module @partial_product_zext(in %a : i3, in %b : i3, out sum : i6) {
  %c0_i3 = hw.constant 0 : i3
  %0 = comb.concat %c0_i3, %a : i3, i3
  %1 = comb.concat %c0_i3, %b : i3, i3
  %2:3 = datapath.partial_product %0, %1 : (i6, i6) -> (i6, i6, i6)
  %3 = comb.add %2#0, %2#1, %2#2 : i6
  hw.output %3 : i6
}

// RUN: circt-lec.sh %t.mlir %s -c1=partial_product_square -c2=partial_product_square
hw.module @partial_product_square(in %a : i4, out sum : i4) {
  %0:4 = datapath.partial_product %a, %a : (i4, i4) -> (i4, i4, i4, i4)
  %1 = comb.add bin %0#0, %0#1, %0#2, %0#3 : i4
  hw.output %1 : i4
}

// RUN: circt-lec.sh %t.mlir %s -c1=partial_product_square_zext -c2=partial_product_square_zext
hw.module @partial_product_square_zext(in %a : i3, out sum : i6) {
  %c0_i3 = hw.constant 0 : i3
  %0 = comb.concat %c0_i3, %a : i3, i3
  %1:3 = datapath.partial_product %0, %0 : (i6, i6) -> (i6, i6, i6)
  %2 = comb.add %1#0, %1#1, %1#2 : i6
  hw.output %2 : i6
}

// RUN: circt-lec.sh %t.mlir %s -c1=partial_product_sext_3 -c2=partial_product_sext_3
hw.module @partial_product_sext_3(in %a : i3, in %b : i3, out sum : i6) {
  %0 = comb.extract %a from 2 : (i3) -> i1
  %1 = comb.extract %b from 2 : (i3) -> i1
  %2 = comb.replicate %0 : (i1) -> i3
  %3 = comb.replicate %1 : (i1) -> i3
  %4 = comb.concat %2, %a : i3, i3
  %5 = comb.concat %3, %b : i3, i3
  %6:6 = datapath.partial_product %4, %5 : (i6, i6) -> (i6, i6, i6, i6, i6, i6)
  %7 = comb.add %6#0, %6#1, %6#2, %6#3, %6#4, %6#5 : i6
  hw.output %7 : i6
}

// RUN: circt-lec.sh %t.mlir %s -c1=partial_product_sext_4 -c2=partial_product_sext_4
hw.module @partial_product_sext_4(in %a : i4, in %b : i4, out sum : i8) {
  %0 = comb.extract %a from 3 : (i4) -> i1
  %1 = comb.replicate %0 : (i1) -> i4
  %2 = comb.concat %1, %a : i4, i4
  %3 = comb.extract %b from 3 : (i4) -> i1
  %4 = comb.replicate %3 : (i1) -> i4
  %5 = comb.concat %4, %b : i4, i4
  %6:8 = datapath.partial_product %2, %5 : (i8, i8) -> (i8, i8, i8, i8, i8, i8, i8, i8)
  %7 = comb.add %6#0, %6#1, %6#2, %6#3, %6#4, %6#5, %6#6, %6#7 : i8
  hw.output %7 : i8
}   

// RUN: circt-lec.sh %t.mlir %s -c1=pos_partial_product_4 -c2=pos_partial_product_4
hw.module @pos_partial_product_4(in %a : i4, in %b : i4, in %c : i4, out sum : i4) {
  %0:4 = datapath.pos_partial_product %a, %b, %c : (i4, i4, i4) -> (i4, i4, i4, i4)
  %1 = comb.add bin %0#0, %0#1, %0#2, %0#3 : i4
  hw.output %1 : i4
}

// RUN: circt-lec.sh %t.mlir %s -c1=pos_partial_product_zext -c2=pos_partial_product_zext
hw.module @pos_partial_product_zext(in %a : i4, in %b : i3, in %c : i4, out sum : i8) {
  %c0_i4 = hw.constant 0 : i4
  %c0_i5 = hw.constant 0 : i5
  %0 = comb.concat %c0_i4, %a : i4, i4
  %1 = comb.concat %c0_i5, %b : i5, i3
  %2 = comb.concat %c0_i4, %c : i4, i4
  %3:8 = datapath.pos_partial_product %0, %1, %2 : (i8, i8, i8) -> (i8, i8, i8, i8, i8, i8, i8, i8)
  %4 = comb.add %3#0, %3#1, %3#2, %3#3, %3#4, %3#5, %3#6, %3#7 : i8 
  hw.output %4 : i8
}

// RUN: circt-lec.sh %t.mlir %s -c1=compress_3 -c2=compress_3
hw.module @compress_3(in %a : i4, in %b : i4, in %c : i4, out sum : i4) {
  %0:2 = datapath.compress %a, %b, %c : i4 [3 -> 2]
  %1 = comb.add bin %0#0, %0#1 : i4
  hw.output %1 : i4
}

// RUN: circt-lec.sh %t.mlir %s -c1=compress_6 -c2=compress_6
hw.module @compress_6(in %a : i4, in %b : i4, in %c : i4, in %d : i4, in %e : i4, in %f : i4, out sum : i4) {
  %0:3 = datapath.compress %a, %b, %c, %d, %e, %f : i4 [6 -> 3]
  %1 = comb.add bin %0#0, %0#1, %0#2 : i4
  hw.output %1 : i4
}

// RUN: circt-opt %s --pass-pipeline="builtin.module(hw.module(convert-datapath-to-comb{lower-partial-product-to-booth=true lower-compress-to-add=true}))" -o %t.mlir
// RUN: circt-lec.sh %t.mlir %s -c1=partial_product_5 -c2=partial_product_5

// RUN: circt-lec.sh %t.mlir %s -c1=partial_product_4 -c2=partial_product_4

// RUN: circt-lec.sh %t.mlir %s -c1=partial_product_zext -c2=partial_product_zext

// RUN: circt-lec.sh %t.mlir %s -c1=partial_product_sext_3 -c2=partial_product_sext_3

// RUN: circt-lec.sh %t.mlir %s -c1=partial_product_sext_4 -c2=partial_product_sext_4

// RUN: circt-lec.sh %t.mlir %s -c1=compress_3 -c2=compress_3

// RUN: circt-lec.sh %t.mlir %s -c1=compress_6 -c2=compress_6
