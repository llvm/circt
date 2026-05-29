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

// CHECK-LABEL: rtg.sequence @setSelectRandom
rtg.sequence @setSelectRandom(%arg0: i32, %arg1: i32) {
  // CHECK: [[SET:%.+]] = rtg.set_create %arg0, %arg1 : i32
  %set = rtg.set_create %arg0, %arg1 : i32
  // CHECK-NEXT: rtg.set_select_random [[SET]] : !rtg.set<i32>
  // CHECK-NEXT: rtg.set_select_random [[SET]] : !rtg.set<i32>
  // CHECK-NEXT: rtg.set_select_random [[SET]] : !rtg.set<i32>
  // They are not CSE'd and not DCE'd
  %0 = rtg.set_select_random %set : !rtg.set<i32>
  %1 = rtg.set_select_random %set : !rtg.set<i32>
  %2 = rtg.set_select_random %set : !rtg.set<i32>
}

// CHECK-LABEL: rtg.sequence @bagSelectRandom
rtg.sequence @bagSelectRandom(%arg0: i32, %arg1: i32, %arg2: index) {
  // CHECK: [[BAG:%.+]] = rtg.bag_create (%arg2 x %arg0, %arg2 x %arg1) : i32
  %bag = rtg.bag_create (%arg2 x %arg0, %arg2 x %arg1) : i32
  // CHECK-NEXT: rtg.bag_select_random [[BAG]] : !rtg.bag<i32>
  // CHECK-NEXT: rtg.bag_select_random [[BAG]] : !rtg.bag<i32>
  // CHECK-NEXT: rtg.bag_select_random [[BAG]] : !rtg.bag<i32>
  // They are not CSE'd and not DCE'd
  %0 = rtg.bag_select_random %bag : !rtg.bag<i32>
  %1 = rtg.bag_select_random %bag : !rtg.bag<i32>
  %2 = rtg.bag_select_random %bag : !rtg.bag<i32>
}

// CHECK-LABEL: rtg.sequence @randomScope
rtg.sequence @randomScope() {
  // CHECK: rtg.random_scope attributes {a}
  // CHECK: rtg.random_scope attributes {b}
  // CHECK: rtg.random_scope attributes {c}
  // They are not CSE'd and not DCE'd
  rtg.random_scope attributes {a} {}
  rtg.random_scope attributes {b} {}
  rtg.random_scope attributes {c} {}
}

// CHECK-LABEL: rtg.sequence @randomNumberInRange
rtg.sequence @randomNumberInRange() {
  // CHECK: [[LOW:%.+]] = index.constant 0
  // CHECK-NEXT: [[HIGH:%.+]] = index.constant 100
  // CHECK-NEXT: rtg.random_number_in_range {{\[}}[[LOW]], [[HIGH]]{{\]}}
  // CHECK-NEXT: rtg.random_number_in_range {{\[}}[[LOW]], [[HIGH]]{{\]}}
  // CHECK-NEXT: rtg.random_number_in_range {{\[}}[[LOW]], [[HIGH]]{{\]}}
  %low = index.constant 0
  %high = index.constant 100
  // They are not CSE'd and not DCE'd
  %0 = rtg.random_number_in_range [%low, %high]
  %1 = rtg.random_number_in_range [%low, %high]
  %2 = rtg.random_number_in_range [%low, %high]
}
