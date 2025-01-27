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
rtg.sequence @unused() {}

rtg.sequence @seq0(%arg0: index) {
  func.call @dummy2(%arg0) : (index) -> ()
}

rtg.sequence @seq1(%arg0: index) {
  %0 = rtg.get_sequence @seq0 : !rtg.sequence<index>
  %1 = rtg.substitute_sequence %0(%arg0) : !rtg.sequence<index>
  %2 = rtg.randomize_sequence %1
  func.call @dummy2(%arg0) : (index) -> ()
  rtg.embed_sequence %2
  func.call @dummy2(%arg0) : (index) -> ()
}

// CHECK-LABEL: rtg.test @nestedSequences
rtg.test @nestedSequences : !rtg.dict<> {
  // CHECK: index.constant 0
  // CHECK: func.call @dummy2
  // CHECK: func.call @dummy2
  // CHECK: func.call @dummy2
  %0 = index.constant 0
  %1 = rtg.get_sequence @seq1 : !rtg.sequence<index>
  %2 = rtg.substitute_sequence %1(%0) : !rtg.sequence<index>
  %3 = rtg.randomize_sequence %2 
  rtg.embed_sequence %3
}

rtg.sequence @seq2(%arg0: index) {
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
  %2 = rtg.get_sequence @seq2 : !rtg.sequence<index>
  %3 = rtg.substitute_sequence %2(%0) : !rtg.sequence<index>
  %4 = rtg.randomize_sequence %3
  %5 = rtg.get_sequence @seq2 : !rtg.sequence<index>
  %6 = rtg.substitute_sequence %5(%1) : !rtg.sequence<index>
  %7 = rtg.randomize_sequence %6
  rtg.embed_sequence %4
  rtg.embed_sequence %7
}

rtg.sequence @seq3(%arg0: !rtg.set<index>) {
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
  %3 = rtg.get_sequence @seq3 : !rtg.sequence<!rtg.set<index>>
  %4 = rtg.substitute_sequence %3(%2) : !rtg.sequence<!rtg.set<index>>
  %5 = rtg.randomize_sequence %4
  %6 = rtg.get_sequence @seq3 : !rtg.sequence<!rtg.set<index>>
  %7 = rtg.substitute_sequence %6(%2) : !rtg.sequence<!rtg.set<index>>
  %8 = rtg.randomize_sequence %7
  rtg.embed_sequence %5
  rtg.embed_sequence %8
  rtg.embed_sequence %5
}

// CHECK-LABEL: @indexOps
rtg.test @indexOps : !rtg.dict<> {
  // CHECK: [[C:%.+]] = index.constant 2
  %0 = index.constant 1

  // CHECK: func.call @dummy2([[C]])
  %1 = index.add %0, %0
  func.call @dummy2(%1) : (index) -> ()

  // CHECK: [[T:%.+]] = index.bool.constant true
  // CHECK: func.call @dummy5([[T]])
  %2 = index.cmp eq(%0, %0)
  func.call @dummy5(%2) : (i1) -> ()

  // CHECK: [[F:%.+]] = index.bool.constant false
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

// CHECK-LABEL: @scfIf
rtg.test @scfIf : !rtg.dict<> {
  %0 = index.bool.constant true
  %1 = index.bool.constant false

  // Don't elaborate body
  scf.if %1 {
    func.call @dummy5(%0) : (i1) -> ()
    scf.yield
  }

  // Test nested ifs 
  // CHECK-NEXT: [[T:%.+]] = index.bool.constant true
  // CHECK-NEXT: func.call @dummy5([[T]])
  // CHECK-NEXT: [[F:%.+]] = index.bool.constant false
  // CHECK-NEXT: func.call @dummy5([[F]])
  scf.if %0 {
    scf.if %0 {
      scf.if %0 {
        func.call @dummy5(%0) : (i1) -> ()
        scf.yield
      }
      scf.yield
    }
    scf.if %0 {
      func.call @dummy5(%1) : (i1) -> ()
      scf.yield
    }
    scf.yield
  }

  // Return values
  // CHECK-NEXT: [[C1:%.+]] = index.constant 1
  // CHECK-NEXT: func.call @dummy2([[C1]])
  %2 = scf.if %0 -> index {
    %3 = index.constant 1
    scf.yield %3 : index
  } else {
    %3 = index.constant 2
    scf.yield %3 : index
  }
  func.call @dummy2(%2) : (index) -> ()
}

// CHECK-LABEL: @scfFor
rtg.test @scfFor : !rtg.dict<> {
  // CHECK-NEXT: [[C0:%.+]] = index.constant 0
  // CHECK-NEXT: func.call @dummy2([[C0]])
  // CHECK-NEXT: [[C1:%.+]] = index.constant 1
  // CHECK-NEXT: func.call @dummy2([[C1]])
  // CHECK-NEXT: [[C2:%.+]] = index.constant 2
  // CHECK-NEXT: func.call @dummy2([[C2]])
  // CHECK-NEXT: func.call @dummy2([[C2]])
  // CHECK-NEXT: [[C4:%.+]] = index.constant 4
  // CHECK-NEXT: func.call @dummy2([[C4]])
  // CHECK-NEXT: [[C3:%.+]] = index.constant 3
  // CHECK-NEXT: func.call @dummy2([[C3]])
  // CHECK-NEXT: func.call @dummy2([[C3]])
  // CHECK-NEXT: func.call @dummy2([[C0]])
  // CHECK-NEXT: }

  %0 = index.constant 0
  %1 = index.constant 2
  %2 = index.constant 5
  %3 = index.constant 1
  // Three iterations
  %4 = scf.for %i = %0 to %2 step %1 iter_args(%a = %0) -> (index) {
    %5 = index.add %a, %3
    func.call @dummy2(%i) : (index) -> ()
    func.call @dummy2(%5) : (index) -> ()
    scf.yield %5 : index
  }
  func.call @dummy2(%4) : (index) -> ()

  // Zero iterations
  %5 = scf.for %i = %0 to %0 step %1 iter_args(%a = %0) -> (index) {
    %6 = index.add %a, %3
    func.call @dummy2(%a) : (index) -> ()
    func.call @dummy2(%6) : (index) -> ()
    scf.yield %6 : index
  }
  func.call @dummy2(%5) : (index) -> ()
}

// CHECK-LABEL: @fixedRegisters
rtg.test @fixedRegisters : !rtg.dict<> {
  // CHECK-NEXT: [[RA:%.+]] = rtg.fixed_reg #rtgtest.ra
  // CHECK-NEXT: [[SP:%.+]] = rtg.fixed_reg #rtgtest.sp
  // CHECK-NEXT: [[IMM:%.+]] = rtgtest.immediate #rtgtest.imm12<0>
  // CHECK-NEXT: rtgtest.rv32i.jalr [[RA]], [[SP]], [[IMM]]
  %ra = rtg.fixed_reg #rtgtest.ra
  %sp = rtg.fixed_reg #rtgtest.sp
  %imm = rtgtest.immediate #rtgtest.imm12<0>
  rtgtest.rv32i.jalr %ra, %sp, %imm
}

// CHECK-LABEL: @virtualRegisters
rtg.test @virtualRegisters : !rtg.dict<> {
  // CHECK-NEXT: [[R0:%.+]] = rtg.virtual_reg [#rtgtest.a0 : !rtgtest.ireg, #rtgtest.a1 : !rtgtest.ireg]
  // CHECK-NEXT: [[R1:%.+]] = rtg.virtual_reg [#rtgtest.s0 : !rtgtest.ireg, #rtgtest.s1 : !rtgtest.ireg]
  // CHECK-NEXT: [[IMM:%.+]] = rtgtest.immediate #rtgtest.imm12<0>
  // CHECK-NEXT: rtgtest.rv32i.jalr [[R0]], [[R1]], [[IMM]]
  // CHECK-NEXT: rtgtest.rv32i.jalr [[R0]], [[R1]], [[IMM]]
  %r0 = rtg.virtual_reg [#rtgtest.a0, #rtgtest.a1]
  %r1 = rtg.virtual_reg [#rtgtest.s0, #rtgtest.s1]
  %imm = rtgtest.immediate #rtgtest.imm12<0>
  rtgtest.rv32i.jalr %r0, %r1, %imm
  rtgtest.rv32i.jalr %r0, %r1, %imm

  // CHECK-NEXT: [[R2:%.+]] = rtg.virtual_reg [#rtgtest.s0 : !rtgtest.ireg, #rtgtest.s1 : !rtgtest.ireg]
  // CHECK-NEXT: rtgtest.rv32i.jalr [[R0]], [[R2]], [[IMM]]
  // CHECK-NEXT: [[R3:%.+]] = rtg.virtual_reg [#rtgtest.s0 : !rtgtest.ireg, #rtgtest.s1 : !rtgtest.ireg]
  // CHECK-NEXT: rtgtest.rv32i.jalr [[R0]], [[R3]], [[IMM]]
  %0 = index.constant 0
  %1 = index.constant 1
  %2 = index.constant 2
  scf.for %i = %0 to %2 step %1 {
    %r2 = rtg.virtual_reg [#rtgtest.s0, #rtgtest.s1]
    rtgtest.rv32i.jalr %r0, %r2, %imm
  }
}

// CHECK-LABEL: @labels
rtg.test @labels : !rtg.dict<> {
  // CHECK-NEXT: [[L0:%.+]] = rtg.label_decl "label0"
  // CHECK-NEXT: rtg.label local [[L0]]
  // CHECK-NEXT: rtg.label local [[L0]]
  // CHECK-NEXT: [[L1:%.+]] = rtg.label_decl "label1_0_1"
  // CHECK-NEXT: rtg.label local [[L1]]
  // CHECK-NEXT: [[L2:%.+]] = rtg.label_decl "label0_0"
  // CHECK-NEXT: rtg.label local [[L2]]
  // CHECK-NEXT: rtg.label local [[L2]]
  // CHECK-NEXT: [[L3:%.+]] = rtg.label_decl "label0_1"
  // CHECK-NEXT: rtg.label local [[L3]]

  %0 = index.constant 0
  %1 = index.constant 1
  %l0 = rtg.label_decl "label0"
  %l1 = rtg.label_decl "label{{0}}", %0
  %l2 = rtg.label_decl "label{{0}}_{{1}}_{{0}}", %1, %0
  %l3 = rtg.label_unique_decl "label0"
  %l4 = rtg.label_unique_decl "label0"
  rtg.label local %l0
  rtg.label local %l1
  rtg.label local %l2
  rtg.label local %l3
  rtg.label local %l3
  rtg.label local %l4
}

// -----

rtg.test @nestedRegionsNotSupported : !rtg.dict<> {
  // expected-error @below {{ops with nested regions must be elaborated away}}
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
