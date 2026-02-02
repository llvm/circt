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
