// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-opt %s --convert-datapath-to-comb -o %t.mlir
// RUN: circt-lec %t.mlir %s -c1=partial_product_5 -c2=partial_product_5 --shared-libs=%libz3 | FileCheck %s --check-prefix=AND5
// AND5: c1 == c2
hw.module @partial_product_5(in %a : i5, in %b : i5, out sum : i5) {
  %0:5 = datapath.partial_product %a, %b : (i5, i5) -> (i5, i5, i5, i5, i5)
  %1 = comb.add bin %0#0, %0#1, %0#2, %0#3, %0#4 : i5
  hw.output %1 : i5
}

// RUN: circt-lec %t.mlir %s -c1=partial_product_4 -c2=partial_product_4 --shared-libs=%libz3 | FileCheck %s --check-prefix=AND4
// AND4: c1 == c2
hw.module @partial_product_4(in %a : i4, in %b : i4, out sum : i4) {
  %0:4 = datapath.partial_product %a, %b : (i4, i4) -> (i4, i4, i4, i4)
  %1 = comb.add bin %0#0, %0#1, %0#2, %0#3 : i4
  hw.output %1 : i4
}

// RUN: circt-lec %t.mlir %s -c1=compress_3 -c2=compress_3 --shared-libs=%libz3 | FileCheck %s --check-prefix=COMP3
// COMP3: c1 == c2
hw.module @compress_3(in %a : i4, in %b : i4, in %c : i4, out sum : i4) {
  %0:2 = datapath.compress %a, %b, %c : i4 [3 -> 2]
  %1 = comb.add bin %0#0, %0#1 : i4
  hw.output %1 : i4
}

// RUN: circt-lec %t.mlir %s -c1=compress_6 -c2=compress_6 --shared-libs=%libz3 | FileCheck %s --check-prefix=COMP6
// COMP6: c1 == c2
hw.module @compress_6(in %a : i4, in %b : i4, in %c : i4, in %d : i4, in %e : i4, in %f : i4, out sum : i4) {
  %0:3 = datapath.compress %a, %b, %c, %d, %e, %f : i4 [6 -> 3]
  %1 = comb.add bin %0#0, %0#1, %0#2 : i4
  hw.output %1 : i4
}

// RUN: circt-opt %s --pass-pipeline="builtin.module(hw.module(convert-datapath-to-comb{lower-partial-product-to-booth=true lower-compress-to-add=true}))" -o %t.mlir
// RUN: circt-lec %t.mlir %s -c1=partial_product_5 -c2=partial_product_5 --shared-libs=%libz3 | FileCheck %s --check-prefix=BOOTH5
// BOOTH5: c1 == c2

// RUN: circt-lec %t.mlir %s -c1=partial_product_4 -c2=partial_product_4 --shared-libs=%libz3 | FileCheck %s --check-prefix=BOOTH4
// BOOTH4: c1 == c2

// RUN: circt-lec %t.mlir %s -c1=compress_3 -c2=compress_3 --shared-libs=%libz3 | FileCheck %s --check-prefix=COMPADD3
// COMPADD3: c1 == c2

// RUN: circt-lec %t.mlir %s -c1=compress_6 -c2=compress_6 --shared-libs=%libz3 | FileCheck %s --check-prefix=COMPADD6
// COMPADD6: c1 == c2
