// RUN: circt-opt -lowering-options=maximumNumberOfTermsPerExpression=4 --export-verilog %s | FileCheck %s

// CHECK-LABEL: module large_use_in_procedural
hw.module @large_use_in_procedural(%clock: i1, %a: i1) {
  // CHECK: wire [[GEN:long_concat]];
  // CHECK: reg [[REG:.+]];

  // CHECK: assign [[GEN]] = a + a + a + a + a;
  // CHECK: always
  sv.always {
    sv.ifdef.procedural "FOO" {
      // This expression should be hoisted and spilled.
      // If there is a namehint, we should use the name.
      %1 = comb.add %a, %a, %a, %a, %a {sv.namehint = "long_concat"}: i1
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

// CHECK-LABEL: module dont_spill_to_procedural_regions
hw.module @dont_spill_to_procedural_regions(%z: i10) -> () {
  %r1 = sv.reg : !hw.inout<i1>
  %r2 = sv.reg : !hw.inout<i10>
  // CHECK: initial begin
  // CHECK-NEXT:   `ifdef BAR
  // CHECK-NEXT:      r1 <= r2 + r2 + r2 + r2 + r2 == z;
  // CHECK-NEXT:   `endif
  // CHECK-NEXT: end // initial
  sv.initial {
    %x = sv.read_inout %r2: !hw.inout<i10>
    sv.ifdef.procedural "BAR" {
      %2 = comb.add %x, %x, %x, %x, %x : i10
      %3 = comb.icmp eq %2, %z: i10
      sv.passign %r1, %3: i1
    }
  }
  hw.output
}
