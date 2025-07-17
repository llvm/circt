// REQUIRES: libz3
// REQUIRES: circt-lec-jit


// datapath.compress
//  RUN: circt-lec %s -c1=adder -c2=compressor --shared-libs=%libz3 | FileCheck %s --check-prefix=DATPATH_COMPRESS
//  DATPATH_COMPRESS: c1 == c2

hw.module @adder(in %in1: i4, in %in2: i4, in %in3 : i4, out out: i4) {
  %sum = comb.add bin %in1, %in2, %in3 : i4
  hw.output %sum : i4
}

hw.module @compressor(in %in1: i4, in %in2: i4, in %in3 : i4, out out: i4) {
  %comp:2 = datapath.compress %in1, %in2, %in3 : i4 [3 -> 2]
  %sum = comb.add bin %comp#0, %comp#1 : i4
  hw.output %sum : i4
}

// datapath.partial_product
//  RUN: circt-lec %s -c1=multiplier -c2=partial_product --shared-libs=%libz3 | FileCheck %s --check-prefix=DATPATH_PARTIAL_PRODUCT
//  DATPATH_PARTIAL_PRODUCT: c1 == c2

hw.module @multiplier(in %in1: i4, in %in2: i4, out out: i4) {
  %sum = comb.mul bin %in1, %in2 : i4
  hw.output %sum : i4
}

hw.module @partial_product(in %in1: i4, in %in2: i4, out out: i4) {
  %pp:4 = datapath.partial_product %in1, %in2 : (i4, i4) -> (i4, i4, i4, i4)
  %sum = comb.add bin %pp#0, %pp#1, %pp#2, %pp#3 : i4
  hw.output %sum : i4
}
