// RUN: circt-opt --llhd-combine-drives %s | FileCheck %s

// Trivial drive forwarding.
// CHECK-LABEL: @Trivial
hw.module @Trivial() {
}
