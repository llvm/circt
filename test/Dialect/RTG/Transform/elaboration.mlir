// RUN: circt-opt --rtg-elaborate=seed=0 --split-input-file --verify-diagnostics %s | FileCheck %s

func.func @dummy1(%arg0: index, %arg1: index, %arg2: !rtg.set<index>) -> () {return}
func.func @dummy2(%arg0: index) -> () {return}
func.func @dummy3(%arg0: !rtg.sequence) -> () {return}
func.func @dummy4(%arg0: index, %arg1: index, %arg2: !rtg.bag<index>, %arg3: !rtg.bag<index>) -> () {return}
func.func @dummy5(%arg0: i1) -> () {return}

// Test the set operations and passing a sequence to another one via argument
// CHECK-LABEL: rtg.test @setOperations
rtg.test @setOperations : !rtg.dict<> {
  // CHECK-NEXT: [[V0:%.+]] = index.constant 2
  // CHECK-NEXT: [[V1:%.+]] = index.constant 3
  // CHECK-NEXT: [[V2:%.+]] = index.constant 4
  // CHECK-NEXT: [[V3:%.+]] = rtg.set_create [[V1]], [[V2]] : index
  // CHECK-NEXT: func.call @dummy1([[V0]], [[V1]], [[V3]]) :
  // CHECK-NEXT: }
  %0 = index.constant 2
  %1 = index.constant 3
  %2 = index.constant 4
  %3 = index.constant 5
  %set0 = rtg.set_create %0, %1, %0 : index
  %set1 = rtg.set_create %2, %0 : index
  %set = rtg.set_union %set0, %set1 : !rtg.set<index>
  %4 = rtg.set_select_random %set : !rtg.set<index> {rtg.elaboration_custom_seed = 1}
  %new_set = rtg.set_create %3, %4 : index
  %diff = rtg.set_difference %set, %new_set : !rtg.set<index>
  %5 = rtg.set_select_random %diff : !rtg.set<index> {rtg.elaboration_custom_seed = 2}
  func.call @dummy1(%4, %5, %diff) : (index, index, !rtg.set<index>) -> ()
}

// CHECK-LABEL: rtg.test @bagOperations
rtg.test @bagOperations : !rtg.dict<> {
  // CHECK-NEXT: [[V0:%.+]] = index.constant 2
  // CHECK-NEXT: [[V1:%.+]] = index.constant 8
  // CHECK-NEXT: [[V2:%.+]] = index.constant 3
  // CHECK-NEXT: [[V3:%.+]] = index.constant 7
  // CHECK-NEXT: [[V4:%.+]] = rtg.bag_create ([[V1]] x [[V0]], [[V3]] x [[V2]]) : index
  // CHECK-NEXT: [[V5:%.+]] = rtg.bag_create ([[V1]] x [[V0]]) : index
  // CHECK-NEXT: func.call @dummy4([[V0]], [[V0]], [[V4]], [[V5]]) :
  %multiple = index.constant 8
  %seven = index.constant 7
  %one = index.constant 1
  %0 = index.constant 2
  %1 = index.constant 3
  %bag0 = rtg.bag_create (%seven x %0, %multiple x %1) : index
  %bag1 = rtg.bag_create (%one x %0) : index
  %bag = rtg.bag_union %bag0, %bag1 : !rtg.bag<index>
  %2 = rtg.bag_select_random %bag : !rtg.bag<index> {rtg.elaboration_custom_seed = 3}
  %new_bag = rtg.bag_create (%one x %2) : index
  %diff = rtg.bag_difference %bag, %new_bag : !rtg.bag<index>
  %3 = rtg.bag_select_random %diff : !rtg.bag<index> {rtg.elaboration_custom_seed = 4}
  %diff2 = rtg.bag_difference %bag, %new_bag inf : !rtg.bag<index>
  %4 = rtg.bag_select_random %diff2 : !rtg.bag<index> {rtg.elaboration_custom_seed = 5}
  func.call @dummy4(%3, %4, %diff, %diff2) : (index, index, !rtg.bag<index>, !rtg.bag<index>) -> ()
}

// CHECK-LABEL: rtg.test @setSize
rtg.test @setSize : !rtg.dict<> {
  // CHECK-NEXT: [[C:%.+]] = index.constant 1
  // CHECK-NEXT: func.call @dummy2([[C]])
  // CHECK-NEXT: }
  %c5 = index.constant 5
  %set = rtg.set_create %c5 : index
  %size = rtg.set_size %set : !rtg.set<index>
  func.call @dummy2(%size) : (index) -> ()
}

// CHECK-LABEL: rtg.test @bagSize
rtg.test @bagSize : !rtg.dict<> {
  // CHECK-NEXT: [[C:%.+]] = index.constant 1
  // CHECK-NEXT: func.call @dummy2([[C]])
  // CHECK-NEXT: }
  %c8 = index.constant 8
  %c5 = index.constant 5
  %bag = rtg.bag_create (%c8 x %c5) : index
  %size = rtg.bag_unique_size %bag : !rtg.bag<index>
  func.call @dummy2(%size) : (index) -> ()
}

// CHECK-LABEL: @targetTest_target0
// CHECK: [[V0:%.+]] = index.constant 0
// CHECK: func.call @dummy2([[V0]]) :

// CHECK-LABEL: @targetTest_target1
// CHECK: [[V0:%.+]] = index.constant 1
// CHECK: func.call @dummy2([[V0]]) :
rtg.test @targetTest : !rtg.dict<num_cpus: index> {
^bb0(%arg0: index):
  func.call @dummy2(%arg0) : (index) -> ()
}

// CHECK-NOT: @unmatchedTest
rtg.test @unmatchedTest : !rtg.dict<num_cpus: !rtg.sequence> {
^bb0(%arg0: !rtg.sequence):
  func.call @dummy3(%arg0) : (!rtg.sequence) -> ()
}

rtg.target @target0 : !rtg.dict<num_cpus: index> {
  %0 = index.constant 0
  rtg.yield %0 : index
}

rtg.target @target1 : !rtg.dict<num_cpus: index> {
  %0 = index.constant 1
  rtg.yield %0 : index
}

// Unused sequences are removed
// CHECK-NOT: rtg.sequence @unused
rtg.sequence @unused {}

rtg.sequence @seq0 {
^bb0(%arg0: index):
  func.call @dummy2(%arg0) : (index) -> ()
}

rtg.sequence @seq1 {
^bb0(%arg0: index):
  %0 = rtg.sequence_closure @seq0(%arg0 : index)
  func.call @dummy2(%arg0) : (index) -> ()
  rtg.invoke_sequence %0
  func.call @dummy2(%arg0) : (index) -> ()
}

// CHECK-LABEL: rtg.test @nestedSequences
rtg.test @nestedSequences : !rtg.dict<> {
  // CHECK: index.constant 0
  // CHECK: func.call @dummy2
  // CHECK: func.call @dummy2
  // CHECK: func.call @dummy2
  %0 = index.constant 0
  %1 = rtg.sequence_closure @seq1(%0 : index)
  rtg.invoke_sequence %1
}

rtg.sequence @seq2 {
^bb0(%arg0: index):
  func.call @dummy2(%arg0) : (index) -> ()
}

// CHECK-LABEL: rtg.test @sameSequenceDifferentArgs
rtg.test @sameSequenceDifferentArgs : !rtg.dict<> {
  // CHECK: [[C0:%.+]] = index.constant 0
  // CHECK: func.call @dummy2([[C0]])
  // CHECK: [[C1:%.+]] = index.constant 1
  // CHECK: func.call @dummy2([[C1]])
  %0 = index.constant 0
  %1 = index.constant 1
  %2 = rtg.sequence_closure @seq2(%0 : index)
  %3 = rtg.sequence_closure @seq2(%1 : index)
  rtg.invoke_sequence %2
  rtg.invoke_sequence %3
}

rtg.sequence @seq3 {
^bb0(%arg0: !rtg.set<index>):
  %0 = rtg.set_select_random %arg0 : !rtg.set<index> // we can't use a custom seed here because it would render the test useless
  func.call @dummy2(%0) : (index) -> ()
}

// CHECK-LABEL: rtg.test @sequenceClosureFixesRandomization
rtg.test @sequenceClosureFixesRandomization : !rtg.dict<> {
  // CHECK: %idx0 = index.constant 0
  // CHECK: func.call @dummy2(%idx0
  // CHECK: %idx1 = index.constant 1
  // CHECK: func.call @dummy2(%idx1
  // CHECK: func.call @dummy2(%idx0
  %0 = index.constant 0
  %1 = index.constant 1
  %2 = rtg.set_create %0, %1 : index
  %3 = rtg.sequence_closure @seq3(%2 : !rtg.set<index>)
  %4 = rtg.sequence_closure @seq3(%2 : !rtg.set<index>)
  rtg.invoke_sequence %3
  rtg.invoke_sequence %4
  rtg.invoke_sequence %3
}

// CHECK-LABLE: @indexOps
rtg.test @indexOps : !rtg.dict<> {
  // CHECK: [[C:%.+]] = index.constant 2
  // CHECK: [[T:%.+]] = index.bool.constant true
  // CHECK: [[F:%.+]] = index.bool.constant false
  %0 = index.constant 1

  // CHECK: func.call @dummy2([[C]])
  %1 = index.add %0, %0
  func.call @dummy2(%1) : (index) -> ()

  // CHECK: func.call @dummy5([[T]])
  %2 = index.cmp eq(%0, %0)
  func.call @dummy5(%2) : (i1) -> ()

  // CHECK: func.call @dummy5([[F]])
  %3 = index.cmp ne(%0, %0)
  func.call @dummy5(%3) : (i1) -> ()

  // CHECK: func.call @dummy5([[F]])
  %4 = index.cmp ult(%0, %0)
  func.call @dummy5(%4) : (i1) -> ()

  // CHECK: func.call @dummy5([[T]])
  %5 = index.cmp ule(%0, %0)
  func.call @dummy5(%5) : (i1) -> ()

  // CHECK: func.call @dummy5([[F]])
  %6 = index.cmp ugt(%0, %0)
  func.call @dummy5(%6) : (i1) -> ()

  // CHECK: func.call @dummy5([[T]])
  %7 = index.cmp uge(%0, %0)
  func.call @dummy5(%7) : (i1) -> ()

  %8 = index.bool.constant true
  // CHECK: func.call @dummy5([[T]])
  func.call @dummy5(%8) : (i1) -> ()
}

// -----

rtg.test @nestedRegionsNotSupported : !rtg.dict<> {
  // expected-error @below {{nested regions not supported}}
  scf.execute_region { scf.yield }
}

// -----

rtg.test @untypedAttributes : !rtg.dict<> {
  // expected-error @below {{only typed attributes supported for constant-like operations}}
  %0 = rtgtest.constant_test index {value = [10 : index]}
}

// -----

func.func @dummy(%arg0: index) {return}

rtg.test @untypedAttributes : !rtg.dict<> {
  %0 = rtgtest.constant_test index {value = "str"}
  // expected-error @below {{materializer of dialect 'builtin' unable to materialize value for attribute '"str"'}}
  // expected-note @below {{while materializing value for operand#0}}
  func.call @dummy(%0) : (index) -> ()
}
