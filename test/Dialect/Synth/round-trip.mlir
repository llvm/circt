// RUN: circt-opt --verify-roundtrip --verify-diagnostics %s | FileCheck %s
// CHECK-LABEL: @basic
// CHECK: synth.mig.maj_inv
hw.module @basic(in %a : i4, out result : i4) {
    %0 = synth.mig.maj_inv not %a, %a, %a {sv.namehint = "out0"} : i4
    hw.output %0 : i4
}
