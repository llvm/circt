// Test that circt-bmc produces well-formed SMT output for a wrong contract
// (one the solver would find a counterexample for at runtime).

// RUN: circt-bmc %s --module WrongMul_CheckContract_0 -b 1 --emit-mlir | FileCheck %s

// The pipeline should still succeed — it produces valid SMT IR regardless of
// whether the property holds.  The solver (at JIT runtime) determines pass/fail.
// CHECK:       func.func @WrongMul_CheckContract_0()
// CHECK:         smt.solver
// CHECK:       func.func @bmc_circuit(%{{.+}}: !smt.bv<42>)
// CHECK:         smt.assert

// A wrong contract: claims a * 9 but implements a * 8 (missing + a).
hw.module @WrongMul(in %a: i42, out z: i42) {
  %c3_i42 = hw.constant 3 : i42
  %c9_i42 = hw.constant 9 : i42
  %0 = comb.shl %a, %c3_i42 : i42
  %1 = verif.contract %0 : i42 {
    %2 = comb.mul %a, %c9_i42 : i42
    %3 = comb.icmp eq %1, %2 : i42
    verif.ensure %3 : i1
  }
  hw.output %1 : i42
}
