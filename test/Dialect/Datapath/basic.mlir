// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: @compressor
hw.module @compressor(in %a : i4, in %b : i4, in %c : i4, out carry : i4, out save : i4) {
  // CHECK-NEXT: datapath.compress %a, %b, %c : i4 [3 -> 2]
  %0:2 = datapath.compress %a, %b, %c : i4 [3 -> 2]
  hw.output %0#0, %0#1 : i4, i4
}

// CHECK-LABEL: @partial_product
hw.module @partial_product(in %a : i3, in %b : i3, out pp0 : i3, out pp1 : i3, out pp2 : i3) {
  // CHECK-NEXT: datapath.partial_product %a, %b : (i3, i3) -> (i3, i3, i3)
  %0:3 = datapath.partial_product %a, %b : (i3, i3) -> (i3, i3, i3)
  hw.output %0#0, %0#1, %0#2 : i3, i3, i3
}