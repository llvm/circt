// Test that circt-bmc can verify verif.contract ops end-to-end.

// RUN: circt-bmc %s --module Mul9_CheckContract_0 -b 1 --emit-mlir | FileCheck %s

// The full pipeline (lower-contracts -> lower-tests -> flatten-modules ->
// externalize-registers -> lower-to-bmc -> convert-hw-to-smt -> ...) should
// produce a solver invocation with the contract property encoded in SMT.
// CHECK:       func.func @Mul9_CheckContract_0()
// CHECK:         smt.solver
// CHECK:       func.func @bmc_circuit(%{{.+}}: !smt.bv<42>)
// CHECK:         smt.bv.shl
// CHECK:         smt.bv.add
// CHECK:         smt.bv.mul
// CHECK:         smt.eq
// CHECK:         smt.assert

// A correct contract: a * 9 == (a << 3) + a
hw.module @Mul9(in %a: i42, out z: i42) {
  %c3_i42 = hw.constant 3 : i42
  %c9_i42 = hw.constant 9 : i42
  %0 = comb.shl %a, %c3_i42 : i42
  %1 = comb.add %a, %0 : i42
  %2 = verif.contract %1 : i42 {
    %3 = comb.mul %a, %c9_i42 : i42
    %4 = comb.icmp eq %2, %3 : i42
    verif.ensure %4 : i1
  }
  hw.output %2 : i42
}
