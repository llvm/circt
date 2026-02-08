// RUN: circt-opt --cse %s | FileCheck %s

// CHECK-LABEL: rtg.sequence @seq
// CHECK-SAME: attributes {rtg.some_attr} {
rtg.sequence @seq0() attributes {rtg.some_attr} {
  // CHECK: [[STR:%.+]] = rtg.constant "label_string" : !rtg.string
  %str = rtg.constant "label_string" : !rtg.string
  // CHECK-NEXT: rtg.label_unique_decl [[STR]]
  // CHECK-NEXT: rtg.label_unique_decl [[STR]]
  // They are DCE'd but not CSE'd 
  %2 = rtg.label_unique_decl %str
  %3 = rtg.label_unique_decl %str
  %4 = rtg.label_unique_decl %str
  // CHECK-NEXT: rtg.label global
  rtg.label global %2
  rtg.label global %3
}
