// REQUIRES: z3

// RUN: circt-opt %s --canonicalize -o %t.mlir

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

// RUN: circt-lec.sh %t.mlir %s -c1=partial_product_sext_6 -c2=partial_product_sext_6
hw.module @partial_product_sext_6(in %a : i6, in %b : i6, out e : i12) {
  %0 = comb.extract %a from 5 : (i6) -> i1
  %1 = comb.replicate %0 : (i1) -> i6
  %2 = comb.concat %1, %a : i6, i6
  %3 = comb.extract %b from 5 : (i6) -> i1
  %4 = comb.replicate %3 : (i1) -> i6
  %5 = comb.concat %4, %b : i6, i6
  %6:12 = datapath.partial_product %2, %5 : (i12, i12) -> (i12, i12, i12, i12, i12, i12, i12, i12, i12, i12, i12, i12)
  %7 = comb.add %6#0, %6#1, %6#2, %6#3, %6#4, %6#5, %6#6, %6#7, %6#8, %6#9, %6#10, %6#11 : i12 
  hw.output %7 : i12
}

// RUN: circt-lec.sh %t.mlir %s -c1=sext_compress -c2=sext_compress
hw.module @sext_compress(in %a : i8, in %b : i8, in %c : i4, 
                         out sum1 : i8, out sum2 : i8) {
  
  %c-1_i8 = hw.constant -1 : i8
  // compress(a,b, sext(c))
  %0 = comb.extract %c from 3 : (i4) -> i1
  %1 = comb.replicate %0 : (i1) -> i4
  %2 = comb.concat %1, %c : i4, i4
  %3:2 = datapath.compress %a, %b, %2 : i8 [3 -> 2]
  %4 = comb.add %3#0, %3#1 : i8

  // compress(a,b, ~sext(c))
  %5 = comb.xor %2, %c-1_i8 : i8
  %6:2 = datapath.compress %a, %b, %5 : i8 [3 -> 2]
  %7 = comb.add %6#0, %6#1 : i8
  
  hw.output %4, %7 : i8, i8
}

// RUN: circt-lec.sh %t.mlir %s -c1=pos_partial_product_sext -c2=pos_partial_product_sext
hw.module @pos_partial_product_sext(in %a : i5, in %b : i5, in %c : i5, out P : i10) {
  %0 = comb.extract %a from 4 : (i5) -> i1
  %1 = comb.replicate %0 : (i1) -> i5
  %2 = comb.concat %1, %a : i5, i5
  %3 = comb.extract %b from 4 : (i5) -> i1
  %4 = comb.replicate %3 : (i1) -> i5
  %5 = comb.concat %4, %b : i5, i5
  %6 = comb.extract %c from 4 : (i5) -> i1
  %7 = comb.replicate %6 : (i1) -> i5
  %8 = comb.concat %7, %c : i5, i5
  %9:10 = datapath.pos_partial_product %2, %5, %8 : (i10, i10, i10) -> (i10, i10, i10, i10, i10, i10, i10, i10, i10, i10)
  %10 = comb.add %9#0, %9#1, %9#2, %9#3, %9#4, %9#5, %9#6, %9#7, %9#8, %9#9 : i10
  hw.output %10 : i10
}
