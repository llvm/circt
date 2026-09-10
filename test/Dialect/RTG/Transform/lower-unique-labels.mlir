// RUN: circt-opt --rtg-lower-unique-labels --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @labels
rtg.test @labels() {
  // CHECK: [[L0:%.+]] = rtg.constant #rtg.isa.label<"label">
  // CHECK: [[L1:%.+]] = rtg.constant #rtg.isa.label<"label_0">
  // CHECK: [[L2:%.+]] = rtg.constant #rtg.isa.label<"label_1">
  // CHECK: rtg.label local [[L0]]
  // CHECK: rtg.label local [[L1]]
  // CHECK: rtg.label local [[L2]]
  // CHECK: rtg.label local [[L2]]
  %l0 = rtg.constant #rtg.isa.label<"label">
  %str = rtg.constant "label" : !rtg.string
  %l1 = rtg.label_unique_decl %str
  %l2 = rtg.label_unique_decl %str
  rtg.label local %l0
  rtg.label local %l1
  rtg.label local %l2
  rtg.label local %l2
}
