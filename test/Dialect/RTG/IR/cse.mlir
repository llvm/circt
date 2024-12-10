// RUN: circt-opt --cse %s | FileCheck %s

// CHECK-LABEL: rtg.sequence @seq
// CHECK-SAME: attributes {rtg.some_attr} {
rtg.sequence @seq0 attributes {rtg.some_attr} {
  // CHECK-NEXT: arith.constant
  %arg = arith.constant 1 : index
  // CHECK-NEXT: rtg.label_decl "label_string_{0}_{1}", %{{.*}}, %{{.*}}
  // They are CSE'd and DCE'd 
  %0 = rtg.label_decl "label_string_{0}_{1}", %arg, %arg
  %1 = rtg.label_decl "label_string_{0}_{1}", %arg, %arg
  // CHECK-NEXT: rtg.label_unique_decl "label_string"
  // CHECK-NEXT: rtg.label_unique_decl "label_string"
  // They are DCE'd but not CSE'd 
  %2 = rtg.label_unique_decl "label_string"
  %3 = rtg.label_unique_decl "label_string"
  %4 = rtg.label_unique_decl "label_string"
  // CHECK-NEXT: rtg.label global
  rtg.label global %0
  rtg.label global %1
  rtg.label global %2
  rtg.label global %3
}
