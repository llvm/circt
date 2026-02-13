// RUN: circt-opt --rtg-elaborate=seed=0 --split-input-file --verify-diagnostics --mlir-print-debuginfo --mlir-print-local-scope %s | FileCheck %s

func.func @dummy1(%arg0: index, %arg1: index, %arg2: !rtg.set<index>) -> () {return}
func.func @dummy2(%arg0: index) -> () {return}
func.func @dummy3(%arg0: !rtg.sequence) -> () {return}
func.func @dummy4(%arg0: index, %arg1: index, %arg2: !rtg.bag<index>, %arg3: !rtg.bag<index>) -> () {return}
func.func @dummy5(%arg0: i1) -> () {return}
func.func @dummy6(%arg0: !rtg.isa.immediate<2>) -> () {return}
func.func @dummy7(%arg0: !rtg.array<index>) -> () {return}
func.func @dummy8(%arg0: !rtg.tuple<index, index>) -> () {return}
func.func @dummy9(%arg0: !rtg.set<!rtg.tuple<index, i1, !rtgtest.ireg>>) -> () {return}
func.func @dummy10(%arg0: !rtg.set<!rtg.tuple<index>>) -> () {return}
func.func @dummy11(%arg0: !rtg.set<index>) -> () {return}
func.func @dummy12(%arg0: !rtg.bag<index>) -> () {return}
func.func @dummy13(%arg0: !rtg.isa.memory_block<32>) -> () {return}
func.func @dummy14(%arg0: !rtg.isa.memory<32>) -> () {return}
func.func @dummy15(%arg0: !rtg.isa.immediate<32>) -> () {return}
func.func @dummy16(%arg0: !rtg.isa.immediate<12>) -> () {return}
func.func @dummy17(%arg0: !rtg.isa.immediate<3>) -> () {return}
func.func @dummy18(%arg0: !rtg.string) -> () {return}

rtg.target @singletonTarget : !rtg.dict<singleton: index> {
  %0 = index.constant 0
  rtg.yield %0 : index
}

// CHECK-LABEL: @immediates
rtg.test @immediates(singleton = %none: index) {
  // CHECK-NEXT: [[V0:%.+]] = rtg.constant #rtg.isa.immediate<2, -1>
  // CHECK-NEXT: func.call @dummy6([[V0]]) : (!rtg.isa.immediate<2>) -> ()
  %0 = rtg.constant #rtg.isa.immediate<2, -1>
  func.call @dummy6(%0) : (!rtg.isa.immediate<2>) -> ()

  // CHECK-NEXT: [[V1:%.+]] = rtg.constant #rtg.isa.immediate<2, 1>
  // CHECK-NEXT: func.call @dummy6([[V1]]) : (!rtg.isa.immediate<2>) -> ()
  %1 = index.constant 1
  %2 = rtg.isa.int_to_immediate %1 : !rtg.isa.immediate<2>
  func.call @dummy6(%2) : (!rtg.isa.immediate<2>) -> ()
}

// Test the set operations and passing a sequence to another one via argument
// CHECK-LABEL: rtg.test @setOperations
rtg.test @setOperations(singleton = %none: index) {
  // CHECK-NEXT: [[V0:%.+]] = index.constant 2
  // CHECK-NEXT: [[V1:%.+]] = index.constant 3
  // CHECK-NEXT: [[V2:%.+]] = index.constant 4
  // CHECK-NEXT: [[V3:%.+]] = rtg.set_create [[V1]], [[V2]] : index
  // CHECK-NEXT: func.call @dummy1([[V0]], [[V1]], [[V3]]) :
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

  // CHECK-NEXT: [[V4:%.+]] = index.constant 1 
  // CHECK-NEXT: [[V5:%.+]] = rtg.bag_create ([[V4]] x [[V2]], [[V4]] x [[V0]]) : index 
  // CHECK-NEXT: func.call @dummy12([[V5]]) : (!rtg.bag<index>)
  %6 = rtg.set_convert_to_bag %set1 : !rtg.set<index>
  func.call @dummy12(%6) : (!rtg.bag<index>) -> ()
}

// CHECK-LABEL: rtg.test @setCartesianProduct
rtg.test @setCartesianProduct(singleton = %none: index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %0 = rtg.set_create %idx0, %idx1 : index
  %true = index.bool.constant true
  %false = index.bool.constant false
  %1 = rtg.set_create %true, %false : i1
  %s0 = rtg.constant #rtgtest.s0
  %s1 = rtg.constant #rtgtest.s1
  %2 = rtg.set_create %s0, %s1 : !rtgtest.ireg

  // CHECK-DAG: [[IDX1:%.+]] = index.constant 1
  // CHECK-DAG: [[FALSE:%.+]] = index.bool.constant false
  // CHECK-DAG: [[S1:%.+]] = rtg.constant #rtgtest.s1 : !rtgtest.ireg
  // CHECK-DAG: [[T1:%.+]] = rtg.tuple_create [[IDX1]], [[FALSE]], [[S1]] : index, i1, !rtgtest.ireg
  // CHECK-DAG: [[IDX0:%.+]] = index.constant 0
  // CHECK-DAG: [[T2:%.+]] = rtg.tuple_create [[IDX0]], [[FALSE]], [[S1]] : index, i1, !rtgtest.ireg
  // CHECK-DAG: [[TRUE:%.+]] = index.bool.constant true
  // CHECK-DAG: [[T3:%.+]] = rtg.tuple_create [[IDX1]], [[TRUE]], [[S1]] : index, i1, !rtgtest.ireg
  // CHECK-DAG: [[T4:%.+]] = rtg.tuple_create [[IDX0]], [[TRUE]], [[S1]] : index, i1, !rtgtest.ireg
  // CHECK-DAG: [[S0:%.+]] = rtg.constant #rtgtest.s0 : !rtgtest.ireg
  // CHECK-DAG: [[T5:%.+]] = rtg.tuple_create [[IDX1]], [[FALSE]], [[S0]] : index, i1, !rtgtest.ireg
  // CHECK-DAG: [[T6:%.+]] = rtg.tuple_create [[IDX0]], [[FALSE]], [[S0]] : index, i1, !rtgtest.ireg
  // CHECK-DAG: [[T7:%.+]] = rtg.tuple_create [[IDX1]], [[TRUE]], [[S0]] : index, i1, !rtgtest.ireg
  // CHECK-DAG: [[T8:%.+]] = rtg.tuple_create [[IDX0]], [[TRUE]], [[S0]] : index, i1, !rtgtest.ireg
  // CHECK-DAG: [[SET:%.+]] = rtg.set_create [[T1]], [[T2]], [[T3]], [[T4]], [[T5]], [[T6]], [[T7]], [[T8]] : !rtg.tuple<index, i1, !rtgtest.ireg>
  // CHECK-NEXT: func.call @dummy9([[SET]]) : (!rtg.set<!rtg.tuple<index, i1, !rtgtest.ireg>>) -> ()
  %3 = rtg.set_cartesian_product %0, %1, %2 : !rtg.set<index>, !rtg.set<i1>, !rtg.set<!rtgtest.ireg>
  func.call @dummy9(%3) : (!rtg.set<!rtg.tuple<index, i1, !rtgtest.ireg>>) -> ()
  
  // CHECK-NEXT: [[EMPTY:%.+]] = rtg.set_create  : !rtg.tuple<index, i1, !rtgtest.ireg>
  // CHECK-NEXT: func.call @dummy9([[EMPTY]]) : (!rtg.set<!rtg.tuple<index, i1, !rtgtest.ireg>>) -> ()
  %4 = rtg.set_create : !rtgtest.ireg
  %5 = rtg.set_cartesian_product %0, %1, %4 : !rtg.set<index>, !rtg.set<i1>, !rtg.set<!rtgtest.ireg>
  func.call @dummy9(%5) : (!rtg.set<!rtg.tuple<index, i1, !rtgtest.ireg>>) -> ()

  // CHECK-NEXT: [[T9:%.+]] = rtg.tuple_create [[IDX1]] : index
  // CHECK-NEXT: [[T10:%.+]] = rtg.tuple_create [[IDX0]] : index
  // CHECK-NEXT: [[SET2:%.+]] = rtg.set_create [[T9]], [[T10]] : !rtg.tuple<index>
  // CHECK-NEXT: func.call @dummy10([[SET2]]) : (!rtg.set<!rtg.tuple<index>>) -> ()
  %6 = rtg.set_cartesian_product %0 : !rtg.set<index>
  func.call @dummy10(%6) : (!rtg.set<!rtg.tuple<index>>) -> ()
}

// CHECK-LABEL: rtg.test @bagOperations
rtg.test @bagOperations(singleton = %none: index) {
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

  // CHECK-NEXT: [[SET:%.+]] = rtg.set_create [[V0]], [[V2]] :
  // CHECK-NEXT: func.call @dummy11([[SET]])
  %5 = rtg.bag_convert_to_set %bag0 : !rtg.bag<index>
  func.call @dummy11(%5) : (!rtg.set<index>) -> ()
}

// CHECK-LABEL: rtg.test @setSize
rtg.test @setSize(singleton = %none: index) {
  // CHECK-NEXT: [[C:%.+]] = index.constant 1
  // CHECK-NEXT: func.call @dummy2([[C]])
  // CHECK-NEXT: }
  %c5 = index.constant 5
  %set = rtg.set_create %c5 : index
  %size = rtg.set_size %set : !rtg.set<index>
  func.call @dummy2(%size) : (index) -> ()
}

// CHECK-LABEL: rtg.test @bagSize
rtg.test @bagSize(singleton = %none: index) {
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
rtg.test @targetTest(num_cpus = %num_cpus: index) {
  func.call @dummy2(%num_cpus) : (index) -> ()
}

// CHECK-NOT: @unmatchedTest
rtg.test @unmatchedTest(num_cpus = %num_cpus: !rtg.sequence) {
  func.call @dummy3(%num_cpus) : (!rtg.sequence) -> ()
}

rtg.target @target0 : !rtg.dict<num_cpus: index> {
  %0 = index.constant 0
  rtg.yield %0 : index
}

rtg.target @target1 : !rtg.dict<num_cpus: index> {
  %0 = index.constant 1
  rtg.yield %0 : index
}

rtg.sequence @seq0(%arg0: index) {
  func.call @dummy2(%arg0) : (index) -> ()
}

// CHECK-LABEL: rtg.test @sequenceSubstitution
rtg.test @sequenceSubstitution(singleton = %none: index) {
  // CHECK-NEXT: [[V0:%.+]] = rtg.get_sequence @seq0{{.*}} : !rtg.sequence loc
  // CHECK-NEXT: [[V1:%.+]] = rtg.randomize_sequence [[V0]]
  // CHECK-NEXT: rtg.embed_sequence [[V1]]
  %0 = index.constant 0
  %1 = rtg.get_sequence @seq0 : !rtg.sequence<index>
  %2 = rtg.substitute_sequence %1(%0) : !rtg.sequence<index>
  %3 = rtg.randomize_sequence %2 
  rtg.embed_sequence %3
}

// CHECK-LABEL: rtg.test @sameSequenceDifferentArgs
rtg.test @sameSequenceDifferentArgs(singleton = %none: index) {
  // CHECK-NEXT: [[V0:%.*]] = rtg.get_sequence @seq0_1 : !rtg.sequence
  // CHECK-NEXT: [[V1:%.*]] = rtg.randomize_sequence [[V0]]
  // CHECK-NEXT: rtg.embed_sequence [[V1]]
  // CHECK-NEXT: [[V2:%.*]] = rtg.get_sequence @seq0_2 : !rtg.sequence
  // CHECK-NEXT: [[V3:%.*]] = rtg.randomize_sequence [[V2]]
  // CHECK-NEXT: rtg.embed_sequence [[V3]]
  %0 = index.constant 0
  %1 = index.constant 1
  %2 = rtg.get_sequence @seq0 : !rtg.sequence<index>
  %3 = rtg.substitute_sequence %2(%0) : !rtg.sequence<index>
  %4 = rtg.randomize_sequence %3
  %5 = rtg.get_sequence @seq0 : !rtg.sequence<index>
  %6 = rtg.substitute_sequence %5(%1) : !rtg.sequence<index>
  %7 = rtg.randomize_sequence %6
  rtg.embed_sequence %4
  rtg.embed_sequence %7
}

// CHECK-LABEL: rtg.test @sequenceClosureFixesRandomization
rtg.test @sequenceClosureFixesRandomization(singleton = %none: index) {
  // CHECK-NEXT: [[V0:%.+]] = rtg.get_sequence @seq3_0 : !rtg.sequence
  // CHECK-NEXT: [[V1:%.+]] = rtg.randomize_sequence [[V0]]
  // CHECK-NEXT: rtg.embed_sequence [[V1]]
  // CHECK-NEXT: [[V2:%.+]] = rtg.get_sequence @seq3_1 : !rtg.sequence
  // CHECK-NEXT: [[V3:%.+]] = rtg.randomize_sequence [[V2]]
  // CHECK-NEXT: rtg.embed_sequence [[V3]]
  // CHECK-NEXT: rtg.embed_sequence [[V1]]
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

// CHECK: rtg.sequence @seq3_0() {
// CHECK-NEXT:   %idx0 = index.constant 0
// CHECK-NEXT:   func.call @dummy2(%idx0) : (index) -> ()
// CHECK-NEXT: }
// CHECK: rtg.sequence @seq3_1() {
// CHECK-NEXT:   %idx1 = index.constant 1
// CHECK-NEXT:   func.call @dummy2(%idx1) : (index) -> ()
// CHECK-NEXT: }

rtg.sequence @seq3(%arg0: !rtg.set<index>) {
  %0 = rtg.set_select_random %arg0 : !rtg.set<index> // we can't use a custom seed here because it would render the test useless
  func.call @dummy2(%0) : (index) -> ()
}

// CHECK-LABEL: @indexOps
rtg.test @indexOps(singleton = %none: index) {
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
rtg.test @scfIf(singleton = %none: index) {
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
rtg.test @scfFor(singleton = %none: index) {
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
rtg.test @fixedRegisters(singleton = %none: index) {
  // CHECK-NEXT: [[RA:%.+]] = rtg.constant #rtgtest.ra
  // CHECK-NEXT: [[SP:%.+]] = rtg.constant #rtgtest.sp
  // CHECK-NEXT: [[IMM:%.+]] = rtg.constant #rtg.isa.immediate<12, 0>
  // CHECK-NEXT: rtgtest.rv32i.jalr [[RA]], [[SP]], [[IMM]]
  %ra = rtg.constant #rtgtest.ra
  %sp = rtg.constant #rtgtest.sp
  %imm = rtg.constant #rtg.isa.immediate<12, 0>
  rtgtest.rv32i.jalr %ra, %sp, %imm
}

// CHECK-LABEL: @virtualRegisters
rtg.test @virtualRegisters(singleton = %none: index) {
  // CHECK-NEXT: [[R0:%.+]] = rtg.virtual_reg [#rtgtest.a0 : !rtgtest.ireg, #rtgtest.a1 : !rtgtest.ireg] loc("a")
  // CHECK-NEXT: [[R1:%.+]] = rtg.virtual_reg [#rtgtest.s0 : !rtgtest.ireg, #rtgtest.s1 : !rtgtest.ireg] loc("b")
  // CHECK-NEXT: [[IMM:%.+]] = rtg.constant #rtg.isa.immediate<12, 0>
  // CHECK-NEXT: rtgtest.rv32i.jalr [[R0]], [[R1]], [[IMM]]
  // CHECK-NEXT: rtgtest.rv32i.jalr [[R0]], [[R1]], [[IMM]]
  %r0 = rtg.virtual_reg [#rtgtest.a0, #rtgtest.a1] loc("a")
  %r1 = rtg.virtual_reg [#rtgtest.s0, #rtgtest.s1] loc("b")
  %imm = rtg.constant #rtg.isa.immediate<12, 0>
  rtgtest.rv32i.jalr %r0, %r1, %imm
  rtgtest.rv32i.jalr %r0, %r1, %imm

  // CHECK-NEXT: [[R2:%.+]] = rtg.virtual_reg [#rtgtest.s0 : !rtgtest.ireg, #rtgtest.s1 : !rtgtest.ireg] loc("c")
  // CHECK-NEXT: rtgtest.rv32i.jalr [[R0]], [[R2]], [[IMM]]
  // CHECK-NEXT: [[R3:%.+]] = rtg.virtual_reg [#rtgtest.s0 : !rtgtest.ireg, #rtgtest.s1 : !rtgtest.ireg] loc("c")
  // CHECK-NEXT: rtgtest.rv32i.jalr [[R0]], [[R3]], [[IMM]]
  %0 = index.constant 0
  %1 = index.constant 1
  %2 = index.constant 2
  scf.for %i = %0 to %2 step %1 {
    %r2 = rtg.virtual_reg [#rtgtest.s0, #rtgtest.s1] loc("c")
    rtgtest.rv32i.jalr %r0, %r2, %imm
  }
}

// CHECK-LABEL:  rtg.sequence @valuesWithIdentitySeq{{.*}}(%arg0: !rtgtest.ireg {{.*}}, %arg1: !rtgtest.ireg {{.*}}, %arg2: !rtgtest.ireg {{.*}}) {
// CHECK: rtgtest.rv32i.jalr %arg0, %arg0
// CHECK: rtgtest.rv32i.jalr %arg1, %arg2

// CHECK-LABEL:  rtg.test @valuesWithIdentity
// CHECK: [[VREG0:%.+]] = rtg.virtual_reg [#rtgtest.a0 : !rtgtest.ireg, #rtgtest.a1 : !rtgtest.ireg]
// CHECK: [[VREG1:%.+]] = rtg.virtual_reg [#rtgtest.a0 : !rtgtest.ireg, #rtgtest.a1 : !rtgtest.ireg]
// CHECK: rtgtest.rv32i.jalr [[VREG0]], [[VREG1]]
// CHECK: [[VREG2:%.+]] = rtg.virtual_reg [#rtgtest.a0 : !rtgtest.ireg, #rtgtest.a1 : !rtgtest.ireg]
// CHECK: [[V0:%.+]] = rtg.get_sequence @valuesWithIdentitySeq{{.*}} :
// CHECK: rtg.substitute_sequence [[V0]]([[VREG0]], [[VREG2]], [[VREG1]]) :

rtg.sequence @valuesWithIdentitySeq(%imm: !rtg.isa.immediate<12>, %reg: !rtgtest.ireg, %set0: !rtg.set<!rtgtest.ireg>, %set1: !rtg.set<!rtgtest.ireg>) {
  rtgtest.rv32i.jalr %reg, %reg, %imm
  %r0 = rtg.set_select_random %set0 : !rtg.set<!rtgtest.ireg>
  %r1 = rtg.set_select_random %set1 : !rtg.set<!rtgtest.ireg>
  rtgtest.rv32i.jalr %r0, %r1, %imm
}

rtg.test @valuesWithIdentity(singleton = %none: index) {
  %r0 = rtg.virtual_reg [#rtgtest.a0, #rtgtest.a1]
  %r1 = rtg.virtual_reg [#rtgtest.a0, #rtgtest.a1]
  %r2 = rtg.virtual_reg [#rtgtest.a0, #rtgtest.a1]
  %r3 = rtg.virtual_reg [#rtgtest.a0, #rtgtest.a1]
  %imm = rtg.constant #rtg.isa.immediate<12, 0>
  rtgtest.rv32i.jalr %r0, %r3, %imm
  %set0 = rtg.set_create %r1, %r2 : !rtgtest.ireg
  %set1 = rtg.set_create %r3 : !rtgtest.ireg
  %s0 = rtg.get_sequence @valuesWithIdentitySeq : !rtg.sequence<!rtg.isa.immediate<12>, !rtgtest.ireg, !rtg.set<!rtgtest.ireg>, !rtg.set<!rtgtest.ireg>>
  %s1 = rtg.substitute_sequence %s0(%imm, %r0, %set0, %set1) : !rtg.sequence<!rtg.isa.immediate<12>, !rtgtest.ireg, !rtg.set<!rtgtest.ireg>, !rtg.set<!rtgtest.ireg>>
  %s2 = rtg.randomize_sequence %s1
  rtg.embed_sequence %s2
}

// CHECK-LABEL: @labels
rtg.test @labels(singleton = %none: index) {
  // CHECK-NEXT: [[STR:%.+]] = rtg.constant "unique_label" : !rtg.string
  // CHECK-NEXT: [[L0:%.+]] = rtg.label_unique_decl [[STR]] loc("l_loc")
  // CHECK-NEXT: rtg.label local [[L0]]
  // CHECK-NEXT: [[L1:%.+]] = rtg.constant #rtg.isa.label<"label">
  // CHECK-NEXT: rtg.label local [[L1]]
  %0 = rtg.constant "unique_label" : !rtg.string
  %l0 = rtg.label_unique_decl %0 loc("l_loc")
  %l1 = rtg.constant #rtg.isa.label<"label">
  rtg.label local %l0
  rtg.label local %l1

  // CHECK-NEXT: rtg.label local [[L1]]
  %set = rtg.set_create %l1 : !rtg.isa.label
  %l4 = rtg.set_select_random %set : !rtg.set<!rtg.isa.label>
  rtg.label local %l4
}

// CHECK-LABEL: rtg.test @randomIntegers
rtg.test @randomIntegers(singleton = %none: index) {
  %lower = index.constant 5
  %upper = index.constant 9
  %0 = rtg.random_number_in_range [%lower, %upper] {rtg.elaboration_custom_seed=0}
  // CHECK-NEXT: [[V0:%.+]] = index.constant 5
  // CHECK-NEXT: func.call @dummy2([[V0]])
  func.call @dummy2(%0) : (index) -> ()

  %1 = rtg.random_number_in_range [%lower, %upper] {rtg.elaboration_custom_seed=3}
  // CHECK-NEXT: [[V1:%.+]] = index.constant 8
  // CHECK-NEXT: func.call @dummy2([[V1]])
  func.call @dummy2(%1) : (index) -> ()
}

// CHECK-LABEL: rtg.test @contexts_contextCpu
rtg.test @contexts(cpu0 = %cpu0: !rtgtest.cpu, cpu1 = %cpu1: !rtgtest.cpu) {
  // CHECK-NEXT:    [[L0:%.+]] = rtg.constant #rtg.isa.label<"label0">
  // CHECK-NEXT:    rtg.label local [[L0]]
  // CHECK-NEXT:    [[SEQ0:%.+]] = rtg.get_sequence @switchCpuSeq_0 : !rtg.sequence
  // CHECK-NEXT:    [[SEQ1:%.+]] = rtg.randomize_sequence [[SEQ0]]
  // CHECK-NEXT:    rtg.embed_sequence [[SEQ1]]
  // CHECK-NEXT:    [[L1:%.+]] = rtg.constant #rtg.isa.label<"label1">
  // CHECK-NEXT:    rtg.label local [[L1]]
  %0 = rtg.get_sequence @cpuSeq : !rtg.sequence<!rtgtest.cpu>
  %1 = rtg.substitute_sequence %0(%cpu1) : !rtg.sequence<!rtgtest.cpu>
  %l0 = rtg.constant #rtg.isa.label<"label0">
  rtg.label local %l0
  rtg.on_context %cpu0, %1 : !rtgtest.cpu
  %l1 = rtg.constant #rtg.isa.label<"label1">
  rtg.label local %l1
}

rtg.target @contextCpu : !rtg.dict<cpu0: !rtgtest.cpu, cpu1: !rtgtest.cpu> {
  %cpu0 = rtg.constant #rtgtest.cpu<0>
  %cpu1 = rtg.constant #rtgtest.cpu<1>
  %0 = rtg.get_sequence @switchCpuSeq : !rtg.sequence<!rtgtest.cpu, !rtgtest.cpu, !rtg.sequence>
  %1 = rtg.get_sequence @switchNestedCpuSeq : !rtg.sequence<!rtgtest.cpu, !rtgtest.cpu, !rtg.sequence>
  rtg.context_switch #rtg.default : !rtgtest.cpu -> #rtgtest.cpu<0> : !rtgtest.cpu, %0 : !rtg.sequence<!rtgtest.cpu, !rtgtest.cpu, !rtg.sequence>
  rtg.context_switch #rtgtest.cpu<0> : !rtgtest.cpu -> #rtgtest.cpu<1> : !rtgtest.cpu, %1 : !rtg.sequence<!rtgtest.cpu, !rtgtest.cpu, !rtg.sequence>
  rtg.yield %cpu0, %cpu1 : !rtgtest.cpu, !rtgtest.cpu
}

// CHECK:  rtg.sequence @cpuSeq_0() {
// CHECK-NEXT:    [[L2:%.+]] = rtg.constant #rtg.isa.label<"label2">
// CHECK-NEXT:    rtg.label local [[L2]]
// CHECK-NEXT:    [[SEQ2:%.+]] = rtg.get_sequence @switchNestedCpuSeq_0 : !rtg.sequence
// CHECK-NEXT:    [[SEQ3:%.+]] = rtg.randomize_sequence [[SEQ2]]
// CHECK-NEXT:    rtg.embed_sequence [[SEQ3]]
// CHECK-NEXT:    [[L3:%.+]] = rtg.constant #rtg.isa.label<"label3">
// CHECK-NEXT:    rtg.label local [[L3]]
// CHECK-NEXT:  }
rtg.sequence @cpuSeq(%cpu: !rtgtest.cpu) {
  %l2 = rtg.constant #rtg.isa.label<"label2">
  rtg.label local %l2
  %0 = rtg.get_sequence @nestedCpuSeq : !rtg.sequence
  rtg.on_context %cpu, %0 : !rtgtest.cpu
  %l3 = rtg.constant #rtg.isa.label<"label3">
  rtg.label local %l3
}

// CHECK:  rtg.sequence @nestedCpuSeq_0() {
// CHECK-NEXT:    [[L6:%.+]] = rtg.constant #rtg.isa.label<"label4">
// CHECK-NEXT:    rtg.label local [[L6]]
// CHECK-NEXT:  }
rtg.sequence @nestedCpuSeq() {
  %l4 = rtg.constant #rtg.isa.label<"label4">
  rtg.label local %l4
}

// CHECK:  rtg.sequence @switchCpuSeq_0() {
// CHECK-NEXT:    [[L8:%.+]] = rtg.constant #rtg.isa.label<"label5">
// CHECK-NEXT:    rtg.label local [[L8]]
// CHECK-NEXT:    [[SEQ5:%.+]] = rtg.get_sequence @cpuSeq_0 : !rtg.sequence
// CHECK-NEXT:    [[SEQ6:%.+]] = rtg.randomize_sequence [[SEQ5]]
// CHECK-NEXT:    rtg.embed_sequence [[SEQ6]]
// CHECK-NEXT:    [[L9:%.+]] = rtg.constant #rtg.isa.label<"label6">
// CHECK-NEXT:    rtg.label local [[L9]]
// CHECK-NEXT:  }
rtg.sequence @switchCpuSeq(%parent: !rtgtest.cpu, %child: !rtgtest.cpu, %seq: !rtg.sequence) {
  %l5 = rtg.constant #rtg.isa.label<"label5">
  rtg.label local %l5
  %0 = rtg.randomize_sequence %seq
  rtg.embed_sequence %0
  %l6 = rtg.constant #rtg.isa.label<"label6">
  rtg.label local %l6
}

// CHECK:  rtg.sequence @switchNestedCpuSeq_0() {
// CHECK-NEXT:    [[L12:%.+]] = rtg.constant #rtg.isa.label<"label7">
// CHECK-NEXT:    rtg.label local [[L12]]
// CHECK-NEXT:    [[SEQ8:%.+]] = rtg.get_sequence @nestedCpuSeq{{.*}} : !rtg.sequence
// CHECK-NEXT:    [[SEQ9:%.+]] = rtg.randomize_sequence [[SEQ8]]
// CHECK-NEXT:    rtg.embed_sequence [[SEQ9]]
// CHECK-NEXT:    [[L13:%.+]] = rtg.constant #rtg.isa.label<"label8">
// CHECK-NEXT:    rtg.label local [[L13]]
// CHECK-NEXT:  }
rtg.sequence @switchNestedCpuSeq(%parent: !rtgtest.cpu, %child: !rtgtest.cpu, %seq: !rtg.sequence) {
  %l7 = rtg.constant #rtg.isa.label<"label7">
  rtg.label local %l7
  %0 = rtg.randomize_sequence %seq
  rtg.embed_sequence %0
  %l8 = rtg.constant #rtg.isa.label<"label8">
  rtg.label local %l8
}

rtg.target @singleCoreTarget : !rtg.dict<single_core: !rtgtest.cpu> {
  %0 = rtg.get_sequence @switchCpuSeq : !rtg.sequence<!rtgtest.cpu, !rtgtest.cpu, !rtg.sequence>
  %1 = rtg.get_sequence @switchNestedCpuSeq : !rtg.sequence<!rtgtest.cpu, !rtgtest.cpu, !rtg.sequence>
  rtg.context_switch #rtg.any_context : !rtgtest.cpu -> #rtg.any_context : !rtgtest.cpu, %1 : !rtg.sequence<!rtgtest.cpu, !rtgtest.cpu, !rtg.sequence>
  rtg.context_switch #rtg.default : !rtgtest.cpu -> #rtg.any_context : !rtgtest.cpu, %0 : !rtg.sequence<!rtgtest.cpu, !rtgtest.cpu, !rtg.sequence>
  %2 = rtg.constant #rtgtest.cpu<0>
  rtg.yield %2 : !rtgtest.cpu
}

// CHECK-LABEL: rtg.test @anyContextSwitch_singleCoreTarget
rtg.test @anyContextSwitch(single_core = %single_core: !rtgtest.cpu) {
  // CHECK-NEXT: [[V0:%.+]] = rtg.get_sequence @switchCpuSeq{{.*}} : !rtg.sequence
  // CHECK-NEXT: [[V1:%.+]] = rtg.randomize_sequence [[V0]]
  // CHECK-NEXT: rtg.embed_sequence [[V1]]
  // CHECK-NEXT: }
  %0 = rtg.get_sequence @nestedCpuSeq : !rtg.sequence
  rtg.on_context %single_core, %0 : !rtgtest.cpu
}

rtg.sequence @interleaveSequencesSeq0() {
  rtgtest.rv32i.ebreak
  rtgtest.rv32i.ebreak
}

rtg.sequence @interleaveSequencesSeq1() {
  rtgtest.rv32i.ecall
  rtgtest.rv32i.ecall
}

// CHECK-LABEL: rtg.test @interleaveSequences
rtg.test @interleaveSequences(singleton = %none: index) {
  // CHECK-NEXT: [[V0:%.+]] = rtg.get_sequence @interleaveSequencesSeq0_0 : !rtg.sequence
  // CHECK-NEXT: [[V2:%.+]] = rtg.randomize_sequence [[V0]]
  // CHECK-NEXT: [[V1:%.+]] = rtg.get_sequence @interleaveSequencesSeq1_0 : !rtg.sequence
  // CHECK-NEXT: [[V3:%.+]] = rtg.randomize_sequence [[V1]]
  // CHECK-NEXT: [[V4:%.+]] = rtg.interleave_sequences [[V2]], [[V3]] batch 2
  // CHECK-NEXT: [[V5:%.+]] = rtg.interleave_sequences [[V4]], [[V3]]
  // CHECK-NEXT: [[V6:%.+]] = rtg.interleave_sequences [[V5]], [[V2]]
  // CHECK-NEXT: rtg.embed_sequence [[V6]]
  %0 = rtg.get_sequence @interleaveSequencesSeq0 : !rtg.sequence
  %1 = rtg.get_sequence @interleaveSequencesSeq1 : !rtg.sequence
  %2 = rtg.randomize_sequence %0
  %3 = rtg.randomize_sequence %1
  %4 = rtg.interleave_sequences %2, %3 batch 2
  %5 = rtg.interleave_sequences %4, %3
  %6 = rtg.interleave_sequences %5, %2
  rtg.embed_sequence %6
}

// CHECK-LABEL: rtg.test @arrays
rtg.test @arrays(singleton = %none: index) {
  // CHECK-NEXT: [[V0:%.+]] = rtg.array_create : index
  // CHECK-NEXT: func.call @dummy7([[V0]]) : (!rtg.array<index>) -> ()
  %0 = rtg.array_create : index
  func.call @dummy7(%0) : (!rtg.array<index>) -> ()

  // CHECK-NEXT: [[IDX2:%.+]] = index.constant 2
  // CHECK-NEXT: func.call @dummy2([[IDX2]]) : (index) -> ()
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  %1 = rtg.array_create %idx1, %idx2 : index
  %2 = rtg.array_extract %1[%idx1] : !rtg.array<index>
  func.call @dummy2(%2) : (index) -> ()

  // CHECK-NEXT: func.call @dummy2([[IDX2]]) : (index) -> ()
  %3 = rtg.array_inject %1[%idx1], %idx2 : !rtg.array<index>
  %4 = rtg.array_extract %3[%idx1] : !rtg.array<index>
  func.call @dummy2(%4) : (index) -> ()

  // CHECK-NEXT: func.call @dummy2([[IDX2]]) : (index) -> ()
  %5 = rtg.array_size %3 : !rtg.array<index>
  func.call @dummy2(%5) : (index) -> ()

  // CHECK-NEXT: [[IDX3:%.+]] = index.constant 3
  // CHECK-NEXT: [[V6:%.+]] = rtg.array_create [[IDX3]], [[IDX3]] : index
  // CHECK-NEXT: func.call @dummy7([[V6]]) : (!rtg.array<index>) -> ()
  %idx3 = index.constant 3
  %6 = rtg.array_create %idx3 : index
  %7 = rtg.array_append %6, %idx3 : !rtg.array<index>
  func.call @dummy7(%7) : (!rtg.array<index>) -> ()
}

// CHECK-LABEL: rtg.test @arithOps
rtg.test @arithOps(singleton = %none: index) {
  // CHECK-NEXT: [[V0:%.+]] = index.constant 6
  // CHECK-NEXT: func.call @dummy2([[V0]])

  %0 = arith.constant 3 : index
  %1 = arith.constant true
  %2 = arith.addi %0, %0 : index
  %3 = arith.andi %1, %1 : i1
  %4 = arith.xori %3, %1 : i1
  %5 = arith.ori %4, %1 : i1
  %6 = arith.select %5, %2, %0 : index
  func.call @dummy2(%6) : (index) -> ()
}

// CHECK-LABEL: rtg.test @tuples
rtg.test @tuples(singleton = %none: index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %0 = rtg.tuple_create %idx1, %idx0 : index, index
  %1 = rtg.tuple_extract %0 at 1 : !rtg.tuple<index, index>

  // CHECK-NEXT: %idx1 = index.constant 1
  // CHECK-NEXT: %idx0 = index.constant 0
  // CHECK-NEXT: [[V0:%.+]] = rtg.tuple_create %idx1, %idx0 : index, index
  // CHECK-NEXT: func.call @dummy8([[V0]])
  func.call @dummy8(%0) : (!rtg.tuple<index, index>) -> ()

  // CHECK-NEXT: func.call @dummy2(%idx0)
  func.call @dummy2(%1) : (index) -> ()
}

// CHECK-LABEL: rtg.test @useFolders_singleCoreTarget
rtg.test @useFolders(single_core = %single_core: !rtgtest.cpu) {
  // CHECK-NEXT: index.constant 0
  // CHECK-NEXT: call @dummy2
  %0 = rtgtest.get_hartid %single_core
  func.call @dummy2(%0) : (index) -> ()
}

// CHECK-LABEL: rtg.test @comments
rtg.test @comments() {
  // CHECK-NEXT: [[STR:%.+]] = rtg.constant "hello" : !rtg.string
  %str = rtg.constant "hello" : !rtg.string
  // CHECK-NEXT: rtg.comment [[STR]]
  rtg.comment %str
}

// CHECK-LABEL: rtg.target @memoryBlocks
rtg.target @memoryBlocks : !rtg.dict<mem_block: !rtg.isa.memory_block<32>> {
  // CHECK-NEXT: rtg.isa.memory_block_declare [0x0 - 0x8] : !rtg.isa.memory_block<32> loc("m_b")
  %0 = rtg.isa.memory_block_declare [0x0 - 0x8] : !rtg.isa.memory_block<32> loc("m_b")
  rtg.yield %0 : !rtg.isa.memory_block<32>
}

// CHECK-LABEL: @memoryBlockTest_memoryBlocks
rtg.test @memoryBlockTest(mem_block = %arg0: !rtg.isa.memory_block<32>) {
  func.call @dummy13(%arg0) : (!rtg.isa.memory_block<32>) -> ()
  // CHECK-NEXT: func.call @dummy13(%mem_block)

  // CHECK-NEXT: [[IDX8:%.+]] = index.constant 8 
  // CHECK-NEXT: [[IDX4:%.+]] = index.constant 4
  // CHECK-NEXT: [[MEM:%.+]] = rtg.isa.memory_alloc %mem_block, [[IDX8]], [[IDX4]] : !rtg.isa.memory_block<32> loc("m_a")
  // CHECK-NEXT: func.call @dummy14([[MEM]])
  %idx4 = index.constant 4
  %idx8 = index.constant 8 
  %0 = rtg.isa.memory_alloc %arg0, %idx8, %idx4 : !rtg.isa.memory_block<32> loc("m_a")
  func.call @dummy14(%0) : (!rtg.isa.memory<32>) -> ()

  // CHECK-NEXT: func.call @dummy2([[IDX8]])
  %1 = rtg.isa.memory_size %0 : !rtg.isa.memory<32>
  func.call @dummy2(%1) : (index) -> ()

  // CHECK-NEXT: }
}

rtg.target @subtypeTarget : !rtg.dict<a: index, b: index> {
  %0 = index.constant 0
  rtg.yield %0, %0 : index, index 
}

// CHECK: rtg.test @subtypeMatching_subtypeTarget(
rtg.test @subtypeMatching(b = %b: index) {
  func.call @dummy2(%b) : (index) -> ()
}

// CHECK-LABEL: rtg.test @validateOp
rtg.test @validateOp(singleton = %none: index) {
  // CHECK-NEXT: [[V0:%.+]] = rtg.constant #rtgtest.t0
  // CHECK-NEXT: [[V1:%.+]] = rtg.constant #rtg.isa.immediate<32, 0>
  // CHECK-NEXT: [[V2:%.+]] = rtg.validate [[V0]], [[V1]], "some_id" : !rtgtest.ireg -> !rtg.isa.immediate<32>
  // CHECK-NEXT: func.call @dummy15([[V2]])
  %reg = rtg.constant #rtgtest.t0
  %default = rtg.constant #rtg.isa.immediate<32, 0>
  %0 = rtg.validate %reg, %default, "some_id" : !rtgtest.ireg -> !rtg.isa.immediate<32>
  func.call @dummy15(%0) : (!rtg.isa.immediate<32>) -> ()

  // CHECK-NEXT: [[V3:%.+]] = rtg.constant #rtg.isa.immediate<32, 1>
  // CHECK-NEXT: [[V4:%.+]] = rtg.constant #rtg.isa.immediate<32, 2>
  // CHECK-NEXT: [[V5:%.+]], [[V6:%.+]]:2 = rtg.validate [[V0]], [[V1]] ([[V3]], [[V4]] else [[V4]], [[V3]] : !rtg.isa.immediate<32>, !rtg.isa.immediate<32>) : !rtgtest.ireg -> !rtg.isa.immediate<32>
  // CHECK-NEXT: func.call @dummy15([[V5]])
  // CHECK-NEXT: func.call @dummy15([[V6]]#0)
  // CHECK-NEXT: [[V7:%.+]] = rtg.get_sequence @validateSeq
  // CHECK-NEXT: [[V8:%.+]] = rtg.substitute_sequence [[V7]]([[V6]]#1)
  // CHECK-NEXT: [[V9:%.+]] = rtg.randomize_sequence [[V8]]
  // CHECK-NEXT: rtg.embed_sequence [[V9]]
  %v1 = rtg.constant #rtg.isa.immediate<32, 1>
  %v2 = rtg.constant #rtg.isa.immediate<32, 2>
  %1:3 = rtg.validate %reg, %default (%v1, %v2 else %v2, %v1 : !rtg.isa.immediate<32>, !rtg.isa.immediate<32>) : !rtgtest.ireg -> !rtg.isa.immediate<32>
  func.call @dummy15(%1#0) : (!rtg.isa.immediate<32>) -> ()
  func.call @dummy15(%1#1) : (!rtg.isa.immediate<32>) -> ()
  %2 = rtg.get_sequence @validateSeq : !rtg.sequence<!rtg.isa.immediate<32>>
  %3 = rtg.substitute_sequence %2(%1#2) : !rtg.sequence<!rtg.isa.immediate<32>>
  %4 = rtg.randomize_sequence %3
  rtg.embed_sequence %4
}

rtg.sequence @validateSeq(%arg0: !rtg.isa.immediate<32>) {
  func.call @dummy15(%arg0) : (!rtg.isa.immediate<32>) -> ()
}

// CHECK-LABEL: @immediateOps
rtg.test @immediateOps(singleton = %none: index) {
  // CHECK-NEXT: [[V0:%.+]] = rtg.constant #rtg.isa.immediate<12, -81>
  // CHECK-NEXT: func.call @dummy16([[V0]]) : (!rtg.isa.immediate<12>) -> ()
  %0 = rtg.constant #rtg.isa.immediate<4, 15>
  %1 = rtg.constant #rtg.isa.immediate<8, 175>
  %2 = rtg.isa.concat_immediate %0, %1 : !rtg.isa.immediate<4>, !rtg.isa.immediate<8>
  func.call @dummy16(%2) : (!rtg.isa.immediate<12>) -> ()

  // CHECK-NEXT: [[V1:%.+]] = rtg.constant #rtg.isa.immediate<2, -2>
  // CHECK-NEXT: func.call @dummy6([[V1]]) : (!rtg.isa.immediate<2>) -> ()
  %3 = rtg.constant #rtg.isa.immediate<8, 175>
  %4 = rtg.isa.slice_immediate %3 from 4 : !rtg.isa.immediate<8> -> !rtg.isa.immediate<2>
  func.call @dummy6(%4) : (!rtg.isa.immediate<2>) -> ()

  // CHECK: [[V1:%.+]] = rtg.validate {{.*}}, {{.*}}, "val1" :
  // CHECK-NEXT: [[CONCAT:%.+]] = rtg.isa.concat_immediate [[V1]], [[V1]] :
  // CHECK-NEXT: func.call @dummy16([[CONCAT]]) :
  // CHECK-NEXT: [[SLICE:%.+]] = rtg.isa.slice_immediate [[V1]] from 4 :
  // CHECK-NEXT: func.call @dummy6([[SLICE]]) :
  %reg = rtg.constant #rtgtest.t0
  %def1 = rtg.constant #rtg.isa.immediate<6, 0>
  %val1 = rtg.validate %reg, %def1, "val1" : !rtgtest.ireg -> !rtg.isa.immediate<6>
  %concat = rtg.isa.concat_immediate %val1, %val1 : !rtg.isa.immediate<6>, !rtg.isa.immediate<6>
  func.call @dummy16(%concat) : (!rtg.isa.immediate<12>) -> ()
  %slice = rtg.isa.slice_immediate %val1 from 4 : !rtg.isa.immediate<6> -> !rtg.isa.immediate<2>
  func.call @dummy6(%slice) : (!rtg.isa.immediate<2>) -> ()
}

// CHECK-LABEL: rtg.test @testSuccessAndFailure
rtg.test @testSuccessAndFailure(singleton = %none: index) {
  // CHECK-NEXT: rtg.test.success
  rtg.test.success
  // CHECK-NEXT: [[STR:%.+]] = rtg.constant "hello" : !rtg.string
  %str = rtg.constant "hello" : !rtg.string
  // CHECK-NEXT: rtg.test.failure [[STR]]
  rtg.test.failure %str
}

// CHECK-LABEL: rtg.test @setEquivalence
rtg.test @setEquivalence(singleton = %none: index) {
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  %set1 = rtg.set_create %idx1, %idx2 : index
  %set2 = rtg.set_create %idx2, %idx1 : index
  // CHECK: func.call @dummy11([[SET:%.+]]) :
  // CHECK: func.call @dummy11([[SET]]) :
  func.call @dummy11(%set1) : (!rtg.set<index>) -> ()
  func.call @dummy11(%set2) : (!rtg.set<index>) -> ()
}

// CHECK-LABEL: rtg.test @untypedAttributes
rtg.test @untypedAttributes(singleton = %none: index) {
  // CHECK-NEXT: [[V0:%.+]] = rtgtest.constant_test index {value = "str"}
  // CHECK-NEXT: func.call @dummy2([[V0]]) : (index) -> ()
  %0 = rtgtest.constant_test index {value = "str"}
  func.call @dummy2(%0) : (index) -> ()

  // CHECK-NEXT: [[V1:%.+]] = rtgtest.constant_test index {value = [10 : index]}
  // CHECK-NEXT: func.call @dummy2([[V1]]) : (index) -> ()
  %1 = rtgtest.constant_test index {value = [10 : index]}
  func.call @dummy2(%1) : (index) -> ()
}

// CHECK-LABEL: rtg.target @arith
rtg.target @arith : !rtg.dict<a: i32, b: i32, c: i32, d: i32> {
  %0 = arith.constant 3 : i32
  %1 = arith.addi %0, %0 : i32
  %2 = arith.andi %0, %0 : i32
  %3 = arith.xori %0, %0 : i32
  %4 = arith.ori %0, %0 : i32
  // CHECK: [[V1:%.+]] = rtg.constant 6 : i32 
  // CHECK: [[V2:%.+]] = rtg.constant 3 : i32 
  // CHECK: [[V3:%.+]] = rtg.constant 0 : i32 
  // CHECK: rtg.yield [[V1]], [[V2]], [[V3]], [[V2]]
  rtg.yield %1, %2, %3, %4 : i32, i32, i32, i32
}

// CHECK-LABEL: rtg.test @opsHandlingSymbolicOperands
rtg.test @opsHandlingSymbolicOperands(singleton = %none: index) {
  // CHECK-NEXT: [[REG:%.+]] = rtg.constant #rtgtest.t0
  // CHECK-NEXT: [[INIT:%.+]] = rtg.constant #rtg.isa.immediate<32, 0>
  %reg = rtg.constant #rtgtest.t0
  %i1 = index.bool.constant false
  %idx = index.constant 0
  %init = rtg.constant #rtg.isa.immediate<32, 0>
  
  // CHECK-NEXT: [[VAL:%.+]] = rtg.validate [[REG]], [[INIT]], "symbolic_idx" : !rtgtest.ireg -> !rtg.isa.immediate<32> loc("sym_loc")
  %val = rtg.validate %reg, %init, "symbolic_idx" : !rtgtest.ireg -> !rtg.isa.immediate<32> loc("sym_loc")
  
  // CHECK-NEXT: func.call @dummy15([[VAL]])
  %arr = rtg.array_create %val, %init : !rtg.isa.immediate<32>
  %ext1 = rtg.array_extract %arr[%idx] : !rtg.array<!rtg.isa.immediate<32>>
  func.call @dummy15(%ext1) : (!rtg.isa.immediate<32>) -> ()
  
  // CHECK-NEXT: func.call @dummy15([[VAL]])
  %tuple = rtg.tuple_create %val, %init : !rtg.isa.immediate<32>, !rtg.isa.immediate<32>
  %ext2 = rtg.tuple_extract %tuple at 0 : !rtg.tuple<!rtg.isa.immediate<32>, !rtg.isa.immediate<32>>
  func.call @dummy15(%ext2) : (!rtg.isa.immediate<32>) -> ()
  
  // CHECK-NEXT: func.call @dummy15([[VAL]])
  %selected = arith.select %i1, %init, %val : !rtg.isa.immediate<32>
  func.call @dummy15(%selected) : (!rtg.isa.immediate<32>) -> ()
  
  // CHECK-NEXT: func.call @dummy15([[VAL]])
  %arr2 = rtg.array_create %init, %init : !rtg.isa.immediate<32>
  %arr3 = rtg.array_inject %arr2[%idx], %val : !rtg.array<!rtg.isa.immediate<32>>
  %ext3 = rtg.array_extract %arr3[%idx] : !rtg.array<!rtg.isa.immediate<32>>
  func.call @dummy15(%ext3) : (!rtg.isa.immediate<32>) -> ()
}

// CHECK-LABEL: rtg.test @strings
rtg.test @strings(singleton = %none: index) {
  %0 = rtg.constant "hello" : !rtg.string
  %c4 = index.constant 4
  %c5 = index.constant 5
  %1 = rtg.random_number_in_range [%c4, %c5] {rtg.elaboration_custom_seed=0}
  %3 = rtg.int_format %1
  %5 = rtg.string_concat %0, %3
  // CHECK-NEXT: [[V0:%.+]] = rtg.constant "hello4" : !rtg.string
  // CHECK-NEXT: func.call @dummy18([[V0]])
  func.call @dummy18(%5) : (!rtg.string) -> ()
}

// -----

rtg.target @singletonTarget : !rtg.dict<singleton: index> {
  %0 = index.constant 0
  rtg.yield %0 : index
}

rtg.test @nestedRegionsNotSupported(singleton = %none: index) {
  // expected-error @below {{ops with nested regions must be elaborated away}}
  scf.execute_region { scf.yield }
}

// -----

rtg.target @singletonTarget : !rtg.dict<singleton: index> {
  %0 = index.constant 0
  rtg.yield %0 : index
}

func.func @dummy2(%arg0: index) -> () {return}

rtg.test @randomIntegers(singleton = %none: index) {
  %c4 = index.constant 4
  %c5 = index.constant 5
  // expected-error @below {{cannot select a number from an empty range}}
  %0 = rtg.random_number_in_range [%c5, %c4]
  func.call @dummy2(%0) : (index) -> ()
}

// -----

rtg.sequence @seq0(%seq: !rtg.randomized_sequence) {
  // expected-error @below {{attempting to place sequence derived from seq1 under context #rtgtest.cpu<0> : !rtgtest.cpu, but it was previously randomized for context 'default'}}
  rtg.embed_sequence %seq
}
rtg.sequence @seq1() { }
rtg.sequence @seq(%arg0: !rtgtest.cpu, %arg1: !rtgtest.cpu, %seq: !rtg.sequence) {
  %0 = rtg.randomize_sequence %seq
  rtg.embed_sequence %0
}

rtg.target @invalidRandomizationTarget : !rtg.dict<cpu: !rtgtest.cpu> {
  %0 = rtg.get_sequence @seq : !rtg.sequence<!rtgtest.cpu, !rtgtest.cpu, !rtg.sequence>
  rtg.context_switch #rtg.default : !rtgtest.cpu -> #rtgtest.cpu<0>, %0 : !rtg.sequence<!rtgtest.cpu, !rtgtest.cpu, !rtg.sequence>
  %1 = rtg.constant #rtgtest.cpu<0>
  rtg.yield %1 : !rtgtest.cpu
}

rtg.test @invalidRandomization(cpu = %cpu: !rtgtest.cpu) {
  %0 = rtg.get_sequence @seq1 : !rtg.sequence
  %1 = rtg.randomize_sequence %0
  %2 = rtg.get_sequence @seq0 : !rtg.sequence<!rtg.randomized_sequence>
  %3 = rtg.substitute_sequence %2(%1) : !rtg.sequence<!rtg.randomized_sequence>
  rtg.on_context %cpu, %3 : !rtgtest.cpu
}

// -----

rtg.sequence @seq() {}

rtg.target @target : !rtg.dict<cpu: !rtgtest.cpu> {
  %0 = rtg.constant #rtgtest.cpu<0>
  rtg.yield %0 : !rtgtest.cpu
}

rtg.test @contextSwitchNotAvailable(cpu = %cpu: !rtgtest.cpu) {
  %0 = rtg.get_sequence @seq : !rtg.sequence
  // expected-error @below {{no context transition registered to switch from #rtg.default : !rtgtest.cpu to #rtgtest.cpu<0> : !rtgtest.cpu}}
  rtg.on_context %cpu, %0 : !rtgtest.cpu
}

// -----

rtg.target @singletonTarget : !rtg.dict<singleton: index> {
  %0 = index.constant 0
  rtg.yield %0 : index
}

rtg.test @emptySetSelect(singleton = %none: index) {
  %0 = rtg.set_create : !rtg.isa.label
  // expected-error @below {{cannot select from an empty set}}
  %1 = rtg.set_select_random %0 : !rtg.set<!rtg.isa.label>
  rtg.label local %1
}

// -----

rtg.target @singletonTarget : !rtg.dict<singleton: index> {
  %0 = index.constant 0
  rtg.yield %0 : index
}

rtg.test @emptyBagSelect(singleton = %none: index) {
  %0 = rtg.bag_create : !rtg.isa.label
  // expected-error @below {{cannot select from an empty bag}}
  %1 = rtg.bag_select_random %0 : !rtg.bag<!rtg.isa.label>
  rtg.label local %1
}

// -----

rtg.target @singletonTarget : !rtg.dict<singleton: index> {
  %0 = index.constant 0
  rtg.yield %0 : index
}

func.func @dummy6(%arg0: !rtg.isa.immediate<2>) -> () {return}

rtg.test @integerTooBig(singleton = %none: index) {
  %1 = index.constant 8
  // expected-error @below {{cannot represent 8 with 2 bits}}
  %2 = rtg.isa.int_to_immediate %1 : !rtg.isa.immediate<2>
  func.call @dummy6(%2) : (!rtg.isa.immediate<2>) -> ()
}

// -----

rtg.target @singletonTarget : !rtg.dict<singleton: index> {
  %0 = index.constant 0
  rtg.yield %0 : index
}

func.func @dummy6(%arg0: index) -> () {return}

rtg.test @oobArrayAccess(singleton = %none: index) {
  %0 = index.constant 0
  %1 = rtg.array_create : index
  // expected-error @below {{invalid to access index 0 of an array with 0 elements}}
  %2 = rtg.array_extract %1[%0] : !rtg.array<index>
  func.call @dummy6(%2) : (index) -> ()
}

// -----

rtg.target @singletonTarget : !rtg.dict<singleton: index> {
  %0 = index.constant 0
  rtg.yield %0 : index
}

func.func @dummy6(%arg0: !rtg.array<index>) -> () {return}

rtg.test @oobArrayAccess(singleton = %none: index) {
  %0 = index.constant 0
  %1 = rtg.array_create : index
  // expected-error @below {{invalid to access index 0 of an array with 0 elements}}
  %2 = rtg.array_inject %1[%0], %0 : !rtg.array<index>
  func.call @dummy6(%2) : (!rtg.array<index>) -> ()
}
