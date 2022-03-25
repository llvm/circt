// RUN: circt-opt -lowering-options=maximumNumberOfTermsPerExpression=4 --export-verilog %s | FileCheck %s

// CHECK-LABEL: module large_use_in_procedural
hw.module @large_use_in_procedural(%clock: i1, %a: i1) {
  // CHECK: wire [[GEN:.+]];
  // CHECK: reg [[REG:.+]];

  // CHECK: assign [[GEN]] = a + a + a + a + a;
  // CHECK: always
  sv.always {
    sv.ifdef.procedural "FOO" {
      // This expression should be hoisted and spilled.
      %1 = comb.add %a, %a, %a, %a, %a : i1
      // CHECK: if ([[GEN]])
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

  %reg = sv.reg : !hw.inout<i1>
  sv.alwaysff(posedge %clock) {
    // CHECK: [[REG]] <= a;
    sv.passign %reg, %a : i1
    %0 = sv.read_inout %reg : !hw.inout<i1>
    // This expression cannot be hoisted, even though it's over the limit.
    %1 = comb.add %0, %0, %0, %0, %0 : i1
    // CHECK: if ([[REG]] + [[REG]] + [[REG]] + [[REG]] + [[REG]])
    sv.if %1 {
      sv.exit
    }
  }
}
