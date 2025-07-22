// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-opt %s --convert-datapath-to-comb -o %t.mlir
// RUN: circt-lec %t.mlir %s -c1=partial_product_6 -c2=partial_product_6 --shared-libs=%libz3 | FileCheck %s --check-prefix=PP_6
// PP_6: c1 == c2
hw.module @partial_product_6(in %a : i6, in %b : i6, out sum : i6) {
  %0:6 = datapath.partial_product %a, %b : (i6, i6) -> (i6, i6, i6, i6, i6, i6)
  %1 = comb.add bin %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 : i6
  hw.output %1 : i6
}

// RUN: circt-lec %t.mlir %s -c1=partial_product_16 -c2=partial_product_16 --shared-libs=%libz3 | FileCheck %s --check-prefix=PP_16
// PP_16: c1 == c2
hw.module @partial_product_16(in %a : i16, in %b : i16, out sum : i16) {
  %0:16 = datapath.partial_product %a, %b : (i16, i16) -> (i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16)
  %1 = comb.add bin %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7, %0#8, %0#9, %0#10, %0#11, %0#12, %0#13, %0#14, %0#15 : i16
  hw.output %1 : i16
}

// RUN: circt-lec %t.mlir %s -c1=partial_product_17 -c2=partial_product_17 --shared-libs=%libz3 | FileCheck %s --check-prefix=PP_17
// PP_17: c1 == c2
hw.module @partial_product_17(in %a : i17, in %b : i17, out sum : i17) {
  %0:17 = datapath.partial_product %a, %b : (i17, i17) -> (i17, i17, i17, i17, i17, i17, i17, i17, i17, i17, i17, i17, i17, i17, i17, i17, i17)
  %1 = comb.add bin %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7, %0#8, %0#9, %0#10, %0#11, %0#12, %0#13, %0#14, %0#15, %0#16 : i17
  hw.output %1 : i17
}
