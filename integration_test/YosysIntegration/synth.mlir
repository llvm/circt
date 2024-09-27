// REQUIRES: libz3
// REQUIRES: circt-lec-jit
// REQUIRES: yosys-integration

// Run synthesis and check the LEC.
// RUN: circt-opt --pass-pipeline='builtin.module(yosys-optimizer{passes=synth},canonicalize)' -o %t.mlir %s

// RUN: circt-lec %s %t.mlir -c1=Arith -c2=Arith --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB
// COMB: c1 == c2

hw.module @Arith(in %in1 : i2, in %in2 : i2, out add : i2, out sub: i2, out mul: i2, out and: i2, out or: i2, out xor: i2 ) {
    %0 = comb.add %in1, %in2: i2
    %1 = comb.sub %in1, %in2: i2
    %2 = comb.mul %in1, %in2: i2
    %3 = comb.and %in1, %in2: i2
    %4 = comb.or %in1, %in2: i2
    %5 = comb.xor %in1, %in2: i2
    hw.output %0, %1, %2, %3, %4, %5 : i2, i2, i2, i2, i2, i2
}

// RUN: circt-lec %s %t.mlir -c1=ICmp -c2=ICmp --shared-libs=%libz3 | FileCheck %s --check-prefix=ICMP
// ICMP: c1 == c2
hw.module @ICmp(in %a : i2, in %b : i2, out eq : i1, 
                out ne: i1, out slt: i1, out sle: i1, out sgt: i1, out sge: i1,
                out ult: i1, out ule: i1, out ugt: i1, out uge: i1
                ) {
  %eq = comb.icmp eq %a, %b : i2
  %ne = comb.icmp ne %a, %b : i2
  %slt = comb.icmp slt %a, %b : i2
  %sle = comb.icmp sle %a, %b : i2
  %sgt = comb.icmp sgt %a, %b : i2
  %sge = comb.icmp sge %a, %b : i2
  %ult = comb.icmp ult %a, %b : i2
  %ule = comb.icmp ule %a, %b : i2
  %ugt = comb.icmp ugt %a, %b : i2
  %uge = comb.icmp uge %a, %b : i2
  hw.output %eq, %ne, %slt, %sle, %sgt, %sge, %ult, %ule, %ugt, %uge : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// RUN: circt-lec %s %t.mlir -c1=misc -c2=misc --shared-libs=%libz3 | FileCheck %s --check-prefix=MISC
// MISC: c1 == c2
hw.module @misc(in %cond:i1, in %in1 : i2, in %in2 : i2, in %in3: i5,
                out mux : i2, out extract: i2, out concat: i9, out replicate: i6, out shl: i5, out parity: i1 ) {
  %mux = comb.mux %cond, %in1, %in2 : i2
  %extract = comb.extract %in3 from 3 : (i5) -> i2
  %concat = comb.concat %in1, %in2, %in3 : i2, i2, i5
  %replicate = comb.replicate %in1 : (i2) -> i6
  %shl = comb.shl %in3, %in3 : i5
  %partiy = comb.parity %in3 : i5
  hw.output %mux, %extract, %concat, %replicate, %shl, %partiy: i2, i2, i9, i6, i5,  i1
}

// RUN: circt-lec %s %t.mlir -c1=MultibitMux -c2=MultibitMux --shared-libs=%libz3 | FileCheck %s --check-prefix=MULTIBITMUX
// MULTIBITMUX: c1 == c2
hw.module @MultibitMux(in %a_0 : i3, in %a_1 : i3, in %a_2 : i3, in %a_3 : i3, in %sel : i2, out b : i3) {
  %0 = hw.array_create %a_3, %a_2, %a_1, %a_0 : i3
  %1 = hw.array_get %0[%sel] : !hw.array<4xi3>, i2
  hw.output %1 : i3
}

// These are incorrectly lowered now(LEC failure).
// * comb.shrs
// * hw.array_create + hw.array_get
//   hw.module @MultibitMux(in %a_0 : i1, in %a_1 : i1, in %a_2 : i1, in %sel : i2, out b : i1) {
//     %0 = hw.array_create %a_0, %a_2, %a_1, %a_0 : i1
//     %1 = hw.array_get %0[%sel] : !hw.array<4xi1>, i2
//     hw.output %1 : i1
//   }
