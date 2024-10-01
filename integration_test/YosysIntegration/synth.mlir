// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-opt --pass-pipeline='builtin.module(export-yosys{passes=synth},canonicalize)' -o %t.mlir %s
// RUN: circt-lec %s %t.mlir -c1=Arith -c2=Arith --shared-libs=%libz3 | FileCheck %s --check-prefix=Comb
// RUN: circt-lec %s %t.mlir -c1=ICmp -c2=ICmp --shared-libs=%libz3 | FileCheck %s --check-prefix=ICmp
// Comb: c1 == c2
// ICmp: c1 == c2


hw.module @Arith(in %in1 : i2, in %in2 : i2, out add : i2, out sub: i2, out mul: i2) {
    %0 = comb.add %in1, %in2: i2
    %1 = comb.sub %in1, %in2: i2
    %2 = comb.mul %in1, %in2: i2
    %3 = comb.and %in1, %in2: i2
    %4 = comb.or %in1, %in2: i2
    %5 = comb.xor %in1, %in2: i2
    hw.output %0, %1, %2 : i2, i2, i2
}

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
