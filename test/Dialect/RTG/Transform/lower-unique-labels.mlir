// RUN: circt-opt --rtg-lower-unique-labels --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @labels
rtg.test @labels() {
  // CHECK-NEXT: [[L0:%.+]] = rtg.label_decl "label"
  // CHECK-NEXT: [[L1:%.+]] = rtg.label_decl "label_0"
  // CHECK-NEXT: [[L2:%.+]] = rtg.label_decl "label_1"
  // CHECK-NEXT: rtg.label local [[L0]]
  // CHECK-NEXT: rtg.label local [[L1]]
  // CHECK-NEXT: rtg.label local [[L2]]
  // CHECK-NEXT: rtg.label local [[L2]]
  %l0 = rtg.label_decl "label"
  %l1 = rtg.label_unique_decl "label"
  %l2 = rtg.label_unique_decl "label"
  rtg.label local %l0
  rtg.label local %l1
  rtg.label local %l2
  rtg.label local %l2
}
