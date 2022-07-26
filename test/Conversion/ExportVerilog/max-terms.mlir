// RUN: circt-opt -lowering-options=maximumNumberOfTermsPerExpression=4,disallowLocalVariables --export-verilog %s | FileCheck %s

// CHECK-LABEL: module large_use_in_procedural
hw.module @large_use_in_procedural(%clock: i1, %a: i1) {
  // CHECK-DAG: wire [[GEN_1:long_concat]];
  // CHECK-DAG: wire [[GEN_0:.*]];
  // CHECK-DAG: reg [[REG:.+]];

  // CHECK: assign [[GEN_1]] = a + a + a + a + a;
  // CHECK: always
  sv.always {
    sv.ifdef.procedural "FOO" {
      // This expression should be hoisted and spilled.
      // If there is a namehint, we should use the name.
      %1 = comb.add %a, %a, %a, %a, %a {sv.namehint = "long_concat"}: i1
      // CHECK: if ([[GEN_1]])
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

  // CHECK: assign [[GEN_0]] = reg_0 + reg_0 + reg_0 + reg_0 + reg_0;
  sv.alwaysff(posedge %clock) {
    // CHECK: always
    // CHECK: [[REG]] = a;
    sv.bpassign %reg, %a : i1
    %0 = sv.read_inout %reg : !hw.inout<i1>
    %1 = comb.add %0, %0, %0, %0, %0 : i1
    // CHECK: if ([[GEN_0]])
    sv.if %1 {
      sv.exit
    }
  }
}

// CHECK-LABEL: module large_use_in_procedural_successive
hw.module @large_use_in_procedural_successive(%clock: i1, %a: i1) {
  sv.always posedge %clock {
    %0 = comb.and %a, %a, %a, %a, %a : i1
    %1 = comb.and %a, %a, %a, %a, %a : i1
    // CHECK:      assign {{.*}} = a & a & a & a & a;
    // CHECK-NEXT: assign {{.*}} = a & a & a & a & a;
    sv.if %0 {
      sv.exit
    }
    sv.if %1 {
      sv.exit
    }
  }
}
