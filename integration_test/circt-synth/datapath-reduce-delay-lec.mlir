// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-opt %s --datapath-reduce-delay -o %t.mlir
// RUN: circt-lec %t.mlir %s -c1=add_compare -c2=add_compare --shared-libs=%libz3 | FileCheck %s --check-prefix=COMPARE
// COMPARE: c1 == c2
hw.module @add_compare(in %a : i16, in %b : i16, in %c : i16, out ugt : i1, out uge : i1, out ult : i1, out ule : i1) {
  %false = hw.constant false
  %0 = comb.concat %false, %a : i1, i16
  %1 = comb.concat %false, %b : i1, i16
  %2 = comb.add %0, %1 {comb.nuw} : i17 
  %3 = comb.concat %false, %c : i1, i16
  // Check that we don't apply transformation when overflow is possible
  %4 = comb.add %0, %3 : i17
  %ugt = comb.icmp ugt %2, %3 : i17
  %uge = comb.icmp uge %2, %3 : i17
  %ult = comb.icmp ult %1, %4 : i17
  %ule = comb.icmp ule %1, %4 : i17
  hw.output %ugt, %uge, %ult, %ule : i1, i1, i1, i1
}

// RUN: circt-lec %t.mlir %s -c1=add_mux -c2=add_mux --shared-libs=%libz3 | FileCheck %s --check-prefix=ADDMUX
// ADDMUX: c1 == c2
hw.module @add_mux(in %a : i4, in %b : i4, in %c : i4, in %d : i4, in %e : i4, in %sel : i1, out res : i4) {
  %0 = comb.add %a, %b : i4
  %1 = comb.add %c, %d, %e : i4
  %2 = comb.mux %sel, %0, %1 : i4
  %3 = comb.add %2, %e : i4
  hw.output %3 : i4
}

// RUN: circt-lec %t.mlir %s -c1=fold_adds -c2=fold_adds --shared-libs=%libz3 | FileCheck %s --check-prefix=FOLDADD
// FOLDADD: c1 == c2
hw.module @fold_adds(in %a : i4, in %b : i4, in %c : i4, in %d : i4, out abc : i4, out abd : i4) {
  %0 = comb.add %a, %b : i4
  %1 = comb.add %0, %c : i4
  %2 = comb.add %0, %d : i4
  hw.output %1, %2 : i4, i4
}
