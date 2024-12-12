// RUN: circt-opt --rtg-elaborate=seed=0 --split-input-file --verify-diagnostics %s | FileCheck %s

func.func @dummy1(%arg0: i32, %arg1: i32, %arg2: !rtg.set<i32>) -> () {return}
func.func @dummy2(%arg0: i32) -> () {return}
func.func @dummy3(%arg0: i64) -> () {return}
func.func @dummy4(%arg0: i32, %arg1: i32, %arg2: !rtg.bag<i32>, %arg3: !rtg.bag<i32>) -> () {return}
func.func @dummy5(%arg0: index) -> () {return}

// Test the set operations and passing a sequence to another one via argument
// CHECK-LABEL: rtg.test @setOperations
rtg.test @setOperations : !rtg.dict<> {
  // CHECK-NEXT: [[V0:%.+]] = arith.constant 2 : i32
  // CHECK-NEXT: [[V1:%.+]] = arith.constant 3 : i32
  // CHECK-NEXT: [[V2:%.+]] = arith.constant 4 : i32
  // CHECK-NEXT: [[V3:%.+]] = rtg.set_create [[V1]], [[V2]] : i32
  // CHECK-NEXT: func.call @dummy1([[V0]], [[V1]], [[V3]]) :
  // CHECK-NEXT: }
  %0 = arith.constant 2 : i32
  %1 = arith.constant 3 : i32
  %2 = arith.constant 4 : i32
  %3 = arith.constant 5 : i32
  %set0 = rtg.set_create %0, %1, %0 : i32
  %set1 = rtg.set_create %2, %0 : i32
  %set = rtg.set_union %set0, %set1 : !rtg.set<i32>
  %4 = rtg.set_select_random %set : !rtg.set<i32> {rtg.elaboration_custom_seed = 1}
  %new_set = rtg.set_create %3, %4 : i32
  %diff = rtg.set_difference %set, %new_set : !rtg.set<i32>
  %5 = rtg.set_select_random %diff : !rtg.set<i32> {rtg.elaboration_custom_seed = 2}
  func.call @dummy1(%4, %5, %diff) : (i32, i32, !rtg.set<i32>) -> ()
}

// CHECK-LABEL: rtg.test @bagOperations
rtg.test @bagOperations : !rtg.dict<> {
  // CHECK-NEXT: [[V0:%.+]] = arith.constant 2 : i32
  // CHECK-NEXT: [[V1:%.+]] = arith.constant 8 : index
  // CHECK-NEXT: [[V2:%.+]] = arith.constant 3 : i32
  // CHECK-NEXT: [[V3:%.+]] = arith.constant 7 : index
  // CHECK-NEXT: [[V4:%.+]] = rtg.bag_create ([[V1]] x [[V0]], [[V3]] x [[V2]]) : i32
  // CHECK-NEXT: [[V5:%.+]] = rtg.bag_create ([[V1]] x [[V0]]) : i32
  // CHECK-NEXT: func.call @dummy4([[V0]], [[V0]], [[V4]], [[V5]]) :
  %multiple = arith.constant 8 : index
  %seven = arith.constant 7 : index
  %one = arith.constant 1 : index
  %0 = arith.constant 2 : i32
  %1 = arith.constant 3 : i32
  %bag0 = rtg.bag_create (%seven x %0, %multiple x %1) : i32
  %bag1 = rtg.bag_create (%one x %0) : i32
  %bag = rtg.bag_union %bag0, %bag1 : !rtg.bag<i32>
  %2 = rtg.bag_select_random %bag : !rtg.bag<i32> {rtg.elaboration_custom_seed = 3}
  %new_bag = rtg.bag_create (%one x %2) : i32
  %diff = rtg.bag_difference %bag, %new_bag : !rtg.bag<i32>
  %3 = rtg.bag_select_random %diff : !rtg.bag<i32> {rtg.elaboration_custom_seed = 4}
  %diff2 = rtg.bag_difference %bag, %new_bag inf : !rtg.bag<i32>
  %4 = rtg.bag_select_random %diff2 : !rtg.bag<i32> {rtg.elaboration_custom_seed = 5}
  func.call @dummy4(%3, %4, %diff, %diff2) : (i32, i32, !rtg.bag<i32>, !rtg.bag<i32>) -> ()
}

// CHECK-LABEL: rtg.test @setSize
rtg.test @setSize : !rtg.dict<> {
  // CHECK-NEXT: [[C:%.+]] = arith.constant 1 : index
  // CHECK-NEXT: func.call @dummy5([[C]])
  // CHECK-NEXT: }
  %c5_i32 = arith.constant 5 : i32
  %set = rtg.set_create %c5_i32 : i32
  %size = rtg.set_size %set : !rtg.set<i32>
  func.call @dummy5(%size) : (index) -> ()
}

// CHECK-LABEL: rtg.test @bagSize
rtg.test @bagSize : !rtg.dict<> {
  // CHECK-NEXT: [[C:%.+]] = arith.constant 1 : index
  // CHECK-NEXT: func.call @dummy5([[C]])
  // CHECK-NEXT: }
  %c8 = arith.constant 8 : index
  %c5_i32 = arith.constant 5 : i32
  %bag = rtg.bag_create (%c8 x %c5_i32) : i32
  %size = rtg.bag_unique_size %bag : !rtg.bag<i32>
  func.call @dummy5(%size) : (index) -> ()
}

// CHECK-LABEL: @targetTest_target0
// CHECK: [[V0:%.+]] = arith.constant 0
// CHECK: func.call @dummy2([[V0]]) :

// CHECK-LABEL: @targetTest_target1
// CHECK: [[V0:%.+]] = arith.constant 1
// CHECK: func.call @dummy2([[V0]]) :
rtg.test @targetTest : !rtg.dict<num_cpus: i32> {
^bb0(%arg0: i32):
  func.call @dummy2(%arg0) : (i32) -> ()
}

// CHECK-NOT: @unmatchedTest
rtg.test @unmatchedTest : !rtg.dict<num_cpus: i64> {
^bb0(%arg0: i64):
  func.call @dummy3(%arg0) : (i64) -> ()
}

rtg.target @target0 : !rtg.dict<num_cpus: i32> {
  %0 = arith.constant 0 : i32
  rtg.yield %0 : i32
}

rtg.target @target1 : !rtg.dict<num_cpus: i32> {
  %0 = arith.constant 1 : i32
  rtg.yield %0 : i32
}

// -----

rtg.test @nestedRegionsNotSupported : !rtg.dict<> {
  %cond = arith.constant false
  // expected-error @below {{nested regions not supported}}
  scf.if %cond { }
}

// -----

rtg.test @untypedAttributes : !rtg.dict<> {
  // expected-error @below {{only typed attributes supported for constant-like operations}}
  %0 = rtgtest.constant_test i32 {value = [10 : i32]}
}

// -----

func.func @dummy(%arg0: i32) {return}

rtg.test @untypedAttributes : !rtg.dict<> {
  %0 = rtgtest.constant_test i32 {value = "str"}
  // expected-error @below {{materializer of dialect 'builtin' unable to materialize value for attribute '"str"'}}
  // expected-note @below {{while materializing value for operand#0}}
  func.call @dummy(%0) : (i32) -> ()
}
