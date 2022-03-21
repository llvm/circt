// RUN: circt-opt -lowering-options=maximumNumberOfTermsPerExpression=4 --export-verilog %s | FileCheck %s

// CHECK-LABEL: module large_use_in_procedural
hw.module @large_use_in_procedural(%a: i1) {
  // CHECK: wire _GEN;
  // CHECK: assign _GEN = a + a + a + a + a;
  // CHECK: always
  sv.always {
    sv.ifdef.procedural "FOO" {
      %1 = comb.add %a, %a, %a, %a, %a : i1
      // CHECK: if (_GEN)
      sv.if %1 {
        sv.exit
      }
      %2 = comb.add %a, %a, %a, %a : i1
      // CHECK: if (a + a + a + a)
      sv.if %2 {
        sv.exit
      }
    }
  }
}
