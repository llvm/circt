// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-opt %s --canonicalize -o %t.mlir

// RUN: circt-lec %t.mlir %s -c1=partial_product_sext_3 -c2=partial_product_sext_3 --shared-libs=%libz3 | FileCheck %s --check-prefix=AND3_SEXT
// AND3_SEXT: c1 == c2
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

// RUN: circt-lec %t.mlir %s -c1=partial_product_sext_6 -c2=partial_product_sext_6 --shared-libs=%libz3 | FileCheck %s --check-prefix=AND6_SEXT
// AND6_SEXT: c1 == c2
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

// RUN: circt-lec %t.mlir %s -c1=sext_compress -c2=sext_compress --shared-libs=%libz3 | FileCheck %s --check-prefix=COMP_SEXT
// COMP_SEXT: c1 == c2
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
