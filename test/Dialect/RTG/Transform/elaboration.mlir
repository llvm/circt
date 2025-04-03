// RUN: circt-opt --rtg-elaborate=seed=0 --split-input-file --verify-diagnostics %s | FileCheck %s

func.func @dummy1(%arg0: index, %arg1: index, %arg2: !rtg.set<index>) -> () {return}
func.func @dummy2(%arg0: index) -> () {return}
func.func @dummy3(%arg0: !rtg.sequence) -> () {return}
func.func @dummy4(%arg0: index, %arg1: index, %arg2: !rtg.bag<index>, %arg3: !rtg.bag<index>) -> () {return}
func.func @dummy5(%arg0: i1) -> () {return}
func.func @dummy6(%arg0: !rtg.isa.immediate<2>) -> () {return}
func.func @dummy7(%arg0: !rtg.array<index>) -> () {return}
func.func @dummy8(%arg0: tuple<index, index>) -> () {return}
func.func @dummy9(%arg0: !rtg.set<tuple<index, i1, !rtgtest.ireg>>) -> () {return}
func.func @dummy10(%arg0: !rtg.set<tuple<index>>) -> () {return}
func.func @dummy11(%arg0: !rtg.set<index>) -> () {return}
func.func @dummy12(%arg0: !rtg.bag<index>) -> () {return}

// CHECK-LABEL: @immediates
rtg.test @immediates() {
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
rtg.test @setOperations() {
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
rtg.test @setCartesianProduct() {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %0 = rtg.set_create %idx0, %idx1 : index
  %true = index.bool.constant true
  %false = index.bool.constant false
  %1 = rtg.set_create %true, %false : i1
  %s0 = rtg.fixed_reg #rtgtest.s0
  %s1 = rtg.fixed_reg #rtgtest.s1
  %2 = rtg.set_create %s0, %s1 : !rtgtest.ireg

  // CHECK-DAG: [[IDX1:%.+]] = index.constant 1
  // CHECK-DAG: [[FALSE:%.+]] = index.bool.constant false
  // CHECK-DAG: [[S1:%.+]] = rtg.fixed_reg #rtgtest.s1 : !rtgtest.ireg
  // CHECK-DAG: [[T1:%.+]] = rtg.tuple_create [[IDX1]], [[FALSE]], [[S1]] : index, i1, !rtgtest.ireg
  // CHECK-DAG: [[IDX0:%.+]] = index.constant 0
  // CHECK-DAG: [[T2:%.+]] = rtg.tuple_create [[IDX0]], [[FALSE]], [[S1]] : index, i1, !rtgtest.ireg
  // CHECK-DAG: [[TRUE:%.+]] = index.bool.constant true
  // CHECK-DAG: [[T3:%.+]] = rtg.tuple_create [[IDX1]], [[TRUE]], [[S1]] : index, i1, !rtgtest.ireg
  // CHECK-DAG: [[T4:%.+]] = rtg.tuple_create [[IDX0]], [[TRUE]], [[S1]] : index, i1, !rtgtest.ireg
  // CHECK-DAG: [[S0:%.+]] = rtg.fixed_reg #rtgtest.s0 : !rtgtest.ireg
  // CHECK-DAG: [[T5:%.+]] = rtg.tuple_create [[IDX1]], [[FALSE]], [[S0]] : index, i1, !rtgtest.ireg
  // CHECK-DAG: [[T6:%.+]] = rtg.tuple_create [[IDX0]], [[FALSE]], [[S0]] : index, i1, !rtgtest.ireg
  // CHECK-DAG: [[T7:%.+]] = rtg.tuple_create [[IDX1]], [[TRUE]], [[S0]] : index, i1, !rtgtest.ireg
  // CHECK-DAG: [[T8:%.+]] = rtg.tuple_create [[IDX0]], [[TRUE]], [[S0]] : index, i1, !rtgtest.ireg
  // CHECK-DAG: [[SET:%.+]] = rtg.set_create [[T1]], [[T2]], [[T3]], [[T4]], [[T5]], [[T6]], [[T7]], [[T8]] : tuple<index, i1, !rtgtest.ireg>
  // CHECK-NEXT: func.call @dummy9([[SET]]) : (!rtg.set<tuple<index, i1, !rtgtest.ireg>>) -> ()
  %3 = rtg.set_cartesian_product %0, %1, %2 : !rtg.set<index>, !rtg.set<i1>, !rtg.set<!rtgtest.ireg>
  func.call @dummy9(%3) : (!rtg.set<tuple<index, i1, !rtgtest.ireg>>) -> ()
  
  // CHECK-NEXT: [[EMPTY:%.+]] = rtg.set_create  : tuple<index, i1, !rtgtest.ireg>
  // CHECK-NEXT: func.call @dummy9([[EMPTY]]) : (!rtg.set<tuple<index, i1, !rtgtest.ireg>>) -> ()
  %4 = rtg.set_create : !rtgtest.ireg
  %5 = rtg.set_cartesian_product %0, %1, %4 : !rtg.set<index>, !rtg.set<i1>, !rtg.set<!rtgtest.ireg>
  func.call @dummy9(%5) : (!rtg.set<tuple<index, i1, !rtgtest.ireg>>) -> ()

  // CHECK-NEXT: [[T9:%.+]] = rtg.tuple_create [[IDX1]] : index
  // CHECK-NEXT: [[T10:%.+]] = rtg.tuple_create [[IDX0]] : index
  // CHECK-NEXT: [[SET2:%.+]] = rtg.set_create [[T9]], [[T10]] : tuple<index>
  // CHECK-NEXT: func.call @dummy10([[SET2]]) : (!rtg.set<tuple<index>>) -> ()
  %6 = rtg.set_cartesian_product %0 : !rtg.set<index>
  func.call @dummy10(%6) : (!rtg.set<tuple<index>>) -> ()
}

// CHECK-LABEL: rtg.test @bagOperations
rtg.test @bagOperations() {
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
rtg.test @setSize() {
  // CHECK-NEXT: [[C:%.+]] = index.constant 1
  // CHECK-NEXT: func.call @dummy2([[C]])
  // CHECK-NEXT: }
  %c5 = index.constant 5
  %set = rtg.set_create %c5 : index
  %size = rtg.set_size %set : !rtg.set<index>
  func.call @dummy2(%size) : (index) -> ()
}

// CHECK-LABEL: rtg.test @bagSize
rtg.test @bagSize() {
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
rtg.test @sequenceSubstitution() {
  // CHECK-NEXT: [[V0:%.+]] = rtg.get_sequence @seq0{{.*}} : !rtg.sequence{{$}}
  // CHECK-NEXT: [[V1:%.+]] = rtg.randomize_sequence [[V0]]
  // CHECK-NEXT: rtg.embed_sequence [[V1]]
  %0 = index.constant 0
  %1 = rtg.get_sequence @seq0 : !rtg.sequence<index>
  %2 = rtg.substitute_sequence %1(%0) : !rtg.sequence<index>
  %3 = rtg.randomize_sequence %2 
  rtg.embed_sequence %3
}

// CHECK-LABEL: rtg.test @sameSequenceDifferentArgs
rtg.test @sameSequenceDifferentArgs() {
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
rtg.test @sequenceClosureFixesRandomization() {
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
rtg.test @indexOps() {
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
rtg.test @scfIf() {
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
rtg.test @scfFor() {
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
rtg.test @fixedRegisters() {
  // CHECK-NEXT: [[RA:%.+]] = rtg.fixed_reg #rtgtest.ra
  // CHECK-NEXT: [[SP:%.+]] = rtg.fixed_reg #rtgtest.sp
  // CHECK-NEXT: [[IMM:%.+]] = rtg.constant #rtg.isa.immediate<12, 0>
  // CHECK-NEXT: rtgtest.rv32i.jalr [[RA]], [[SP]], [[IMM]]
  %ra = rtg.fixed_reg #rtgtest.ra
  %sp = rtg.fixed_reg #rtgtest.sp
  %imm = rtg.constant #rtg.isa.immediate<12, 0>
  rtgtest.rv32i.jalr %ra, %sp, %imm
}

// CHECK-LABEL: @virtualRegisters
rtg.test @virtualRegisters() {
  // CHECK-NEXT: [[R0:%.+]] = rtg.virtual_reg [#rtgtest.a0 : !rtgtest.ireg, #rtgtest.a1 : !rtgtest.ireg]
  // CHECK-NEXT: [[R1:%.+]] = rtg.virtual_reg [#rtgtest.s0 : !rtgtest.ireg, #rtgtest.s1 : !rtgtest.ireg]
  // CHECK-NEXT: [[IMM:%.+]] = rtg.constant #rtg.isa.immediate<12, 0>
  // CHECK-NEXT: rtgtest.rv32i.jalr [[R0]], [[R1]], [[IMM]]
  // CHECK-NEXT: rtgtest.rv32i.jalr [[R0]], [[R1]], [[IMM]]
  %r0 = rtg.virtual_reg [#rtgtest.a0, #rtgtest.a1]
  %r1 = rtg.virtual_reg [#rtgtest.s0, #rtgtest.s1]
  %imm = rtg.constant #rtg.isa.immediate<12, 0>
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

// CHECK-LABEL:  rtg.sequence @valuesWithIdentitySeq{{.*}}(%arg0: !rtgtest.ireg, %arg1: !rtgtest.ireg, %arg2: !rtgtest.ireg) {
// CHECK: rtgtest.rv32i.jalr %arg0, %arg0
// CHECK: rtgtest.rv32i.jalr %arg1, %arg2

// CHECK-LABEL:  rtg.test @valuesWithIdentity() {
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

rtg.test @valuesWithIdentity() {
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
rtg.test @labels() {
  // CHECK-NEXT: [[L0:%.+]] = rtg.label_unique_decl "label0"
  // CHECK-NEXT: rtg.label local [[L0]]
  // CHECK-NEXT: [[L1:%.+]] = rtg.label_decl "label0"
  // CHECK-NEXT: rtg.label local [[L1]]
  // CHECK-NEXT: [[L2:%.+]] = rtg.label_decl "label1_0_1"
  // CHECK-NEXT: rtg.label local [[L2]]
  %0 = index.constant 0
  %1 = index.constant 1
  %l0 = rtg.label_unique_decl "label{{0}}", %0
  %l1 = rtg.label_decl "label{{0}}", %0
  %l2 = rtg.label_decl "label{{0}}_{{1}}_{{0}}", %1, %0
  rtg.label local %l0
  rtg.label local %l1
  rtg.label local %l2
}

// CHECK-LABEL: rtg.test @randomIntegers
rtg.test @randomIntegers() {
  %lower = index.constant 5
  %upper = index.constant 10
  %0 = rtg.random_number_in_range [%lower, %upper) {rtg.elaboration_custom_seed=0}
  // CHECK-NEXT: [[V0:%.+]] = index.constant 5
  // CHECK-NEXT: func.call @dummy2([[V0]])
  func.call @dummy2(%0) : (index) -> ()

  %1 = rtg.random_number_in_range [%lower, %upper) {rtg.elaboration_custom_seed=3}
  // CHECK-NEXT: [[V1:%.+]] = index.constant 8
  // CHECK-NEXT: func.call @dummy2([[V1]])
  func.call @dummy2(%1) : (index) -> ()
}

// CHECK-LABEL: rtg.test @contexts_contextCpu
rtg.test @contexts(cpu0 = %cpu0: !rtgtest.cpu, cpu1 = %cpu1: !rtgtest.cpu) {
  // CHECK-NEXT:    [[L0:%.+]] = rtg.label_decl "label0"
  // CHECK-NEXT:    rtg.label local [[L0]]
  // CHECK-NEXT:    [[SEQ0:%.+]] = rtg.get_sequence @switchCpuSeq_0 : !rtg.sequence
  // CHECK-NEXT:    [[SEQ1:%.+]] = rtg.randomize_sequence [[SEQ0]]
  // CHECK-NEXT:    rtg.embed_sequence [[SEQ1]]
  // CHECK-NEXT:    [[L1:%.+]] = rtg.label_decl "label1"
  // CHECK-NEXT:    rtg.label local [[L1]]
  %0 = rtg.get_sequence @cpuSeq : !rtg.sequence<!rtgtest.cpu>
  %1 = rtg.substitute_sequence %0(%cpu1) : !rtg.sequence<!rtgtest.cpu>
  %l0 = rtg.label_decl "label0"
  rtg.label local %l0
  rtg.on_context %cpu0, %1 : !rtgtest.cpu
  %l1 = rtg.label_decl "label1"
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
// CHECK-NEXT:    [[L2:%.+]] = rtg.label_decl "label2"
// CHECK-NEXT:    rtg.label local [[L2]]
// CHECK-NEXT:    [[SEQ2:%.+]] = rtg.get_sequence @switchNestedCpuSeq_0 : !rtg.sequence
// CHECK-NEXT:    [[SEQ3:%.+]] = rtg.randomize_sequence [[SEQ2]]
// CHECK-NEXT:    rtg.embed_sequence [[SEQ3]]
// CHECK-NEXT:    [[L3:%.+]] = rtg.label_decl "label3"
// CHECK-NEXT:    rtg.label local [[L3]]
// CHECK-NEXT:  }
rtg.sequence @cpuSeq(%cpu: !rtgtest.cpu) {
  %l2 = rtg.label_decl "label2"
  rtg.label local %l2
  %0 = rtg.get_sequence @nestedCpuSeq : !rtg.sequence
  rtg.on_context %cpu, %0 : !rtgtest.cpu
  %l3 = rtg.label_decl "label3"
  rtg.label local %l3
}

// CHECK:  rtg.sequence @nestedCpuSeq_0() {
// CHECK-NEXT:    [[L6:%.+]] = rtg.label_decl "label4"
// CHECK-NEXT:    rtg.label local [[L6]]
// CHECK-NEXT:  }
rtg.sequence @nestedCpuSeq() {
  %l4 = rtg.label_decl "label4"
  rtg.label local %l4
}

// CHECK:  rtg.sequence @switchCpuSeq_0() {
// CHECK-NEXT:    [[L8:%.+]] = rtg.label_decl "label5"
// CHECK-NEXT:    rtg.label local [[L8]]
// CHECK-NEXT:    [[SEQ5:%.+]] = rtg.get_sequence @cpuSeq_0 : !rtg.sequence
// CHECK-NEXT:    [[SEQ6:%.+]] = rtg.randomize_sequence [[SEQ5]]
// CHECK-NEXT:    rtg.embed_sequence [[SEQ6]]
// CHECK-NEXT:    [[L9:%.+]] = rtg.label_decl "label6"
// CHECK-NEXT:    rtg.label local [[L9]]
// CHECK-NEXT:  }
rtg.sequence @switchCpuSeq(%parent: !rtgtest.cpu, %child: !rtgtest.cpu, %seq: !rtg.sequence) {
  %l5 = rtg.label_decl "label5"
  rtg.label local %l5
  %0 = rtg.randomize_sequence %seq
  rtg.embed_sequence %0
  %l6 = rtg.label_decl "label6"
  rtg.label local %l6
}

// CHECK:  rtg.sequence @switchNestedCpuSeq_0() {
// CHECK-NEXT:    [[L12:%.+]] = rtg.label_decl "label7"
// CHECK-NEXT:    rtg.label local [[L12]]
// CHECK-NEXT:    [[SEQ8:%.+]] = rtg.get_sequence @nestedCpuSeq{{.*}} : !rtg.sequence
// CHECK-NEXT:    [[SEQ9:%.+]] = rtg.randomize_sequence [[SEQ8]]
// CHECK-NEXT:    rtg.embed_sequence [[SEQ9]]
// CHECK-NEXT:    [[L13:%.+]] = rtg.label_decl "label8"
// CHECK-NEXT:    rtg.label local [[L13]]
// CHECK-NEXT:  }
rtg.sequence @switchNestedCpuSeq(%parent: !rtgtest.cpu, %child: !rtgtest.cpu, %seq: !rtg.sequence) {
  %l7 = rtg.label_decl "label7"
  rtg.label local %l7
  %0 = rtg.randomize_sequence %seq
  rtg.embed_sequence %0
  %l8 = rtg.label_decl "label8"
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

// CHECK-LABEL: @interleaveSequences()
rtg.test @interleaveSequences() {
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
rtg.test @arrays() {
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
}

// CHECK-LABEL: rtg.test @arithOps
rtg.test @arithOps() {
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
rtg.test @tuples() {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %0 = rtg.tuple_create %idx1, %idx0 : index, index
  %1 = rtg.tuple_extract %0 at 1 : tuple<index, index>

  // CHECK-NEXT: %idx1 = index.constant 1
  // CHECK-NEXT: %idx0 = index.constant 0
  // CHECK-NEXT: [[V0:%.+]] = rtg.tuple_create %idx1, %idx0 : index, index
  // CHECK-NEXT: func.call @dummy8([[V0]])
  func.call @dummy8(%0) : (tuple<index, index>) -> ()

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
  // CHECK-NEXT: rtg.comment "this is a comment"
  rtg.comment "this is a comment"
}

rtg.target @memoryBlocks : !rtg.dict<mem_block: !rtg.isa.memory_block<32>> {
  %0 = rtg.isa.memory_block_declare [0x0 - 0x8] : !rtg.isa.memory_block<32>
  rtg.yield %0 : !rtg.isa.memory_block<32>
}

// CHECK-LABEL: @memoryBlockTest_memoryBlocks
rtg.test @memoryBlockTest(mem_block = %arg0: !rtg.isa.memory_block<32>) {
  // CHECK-NEXT: }
}

// -----

rtg.test @nestedRegionsNotSupported() {
  // expected-error @below {{ops with nested regions must be elaborated away}}
  scf.execute_region { scf.yield }
}

// -----

rtg.test @untypedAttributes() {
  // expected-error @below {{only typed attributes supported for constant-like operations}}
  %0 = rtgtest.constant_test index {value = [10 : index]}
}

// -----

func.func @dummy(%arg0: index) {return}

rtg.test @untypedAttributes() {
  %0 = rtgtest.constant_test index {value = "str"}
  // expected-error @below {{materializer of dialect 'builtin' unable to materialize value for attribute '"str"'}}
  // expected-note @below {{while materializing value for operand#0}}
  func.call @dummy(%0) : (index) -> ()
}

// -----

func.func @dummy2(%arg0: index) -> () {return}

rtg.test @randomIntegers() {
  %c5 = index.constant 5
  // expected-error @below {{cannot select a number from an empty range}}
  %0 = rtg.random_number_in_range [%c5, %c5)
  func.call @dummy2(%0) : (index) -> ()
}

// -----

rtg.sequence @seq0(%seq: !rtg.randomized_sequence) {
  // expected-error @below {{attempting to place sequence seq1_0 derived from seq1 under context #rtgtest.cpu<0> : !rtgtest.cpu, but it was previously randomized for context 'default'}}
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

rtg.test @emptySetSelect() {
  %0 = rtg.set_create : !rtg.isa.label
  // expected-error @below {{cannot select from an empty set}}
  %1 = rtg.set_select_random %0 : !rtg.set<!rtg.isa.label>
  rtg.label local %1
}

// -----

rtg.test @emptyBagSelect() {
  %0 = rtg.bag_create : !rtg.isa.label
  // expected-error @below {{cannot select from an empty bag}}
  %1 = rtg.bag_select_random %0 : !rtg.bag<!rtg.isa.label>
  rtg.label local %1
}

// -----

func.func @dummy6(%arg0: !rtg.isa.immediate<2>) -> () {return}

rtg.test @integerTooBig() {
  %1 = index.constant 8
  // expected-error @below {{cannot represent 8 with 2 bits}}
  %2 = rtg.isa.int_to_immediate %1 : !rtg.isa.immediate<2>
  func.call @dummy6(%2) : (!rtg.isa.immediate<2>) -> ()
}

// -----

func.func @dummy6(%arg0: index) -> () {return}

rtg.test @oobArrayAccess() {
  %0 = index.constant 0
  %1 = rtg.array_create : index
  // expected-error @below {{invalid to access index 0 of an array with 0 elements}}
  %2 = rtg.array_extract %1[%0] : !rtg.array<index>
  func.call @dummy6(%2) : (index) -> ()
}

// -----

func.func @dummy6(%arg0: !rtg.array<index>) -> () {return}

rtg.test @oobArrayAccess() {
  %0 = index.constant 0
  %1 = rtg.array_create : index
  // expected-error @below {{invalid to access index 0 of an array with 0 elements}}
  %2 = rtg.array_inject %1[%0], %0 : !rtg.array<index>
  func.call @dummy6(%2) : (!rtg.array<index>) -> ()
}

// -----

rtg.test @arith_invalid_type() {
  %0 = arith.constant 3 : i32
  // expected-error @below {{only index operands supported}}
  %1 = arith.addi %0, %0 : i32
}

// -----

rtg.test @arith_invalid_type() {
  %0 = arith.constant 3 : i32
  // expected-error @below {{only 'i1' operands supported}}
  %1 = arith.andi %0, %0 : i32
}

// -----

rtg.test @arith_invalid_type() {
  %0 = arith.constant 3 : i32
  // expected-error @below {{only 'i1' operands supported}}
  %1 = arith.xori %0, %0 : i32
}

// -----

rtg.test @arith_invalid_type() {
  %0 = arith.constant 3 : i32
  // expected-error @below {{only 'i1' operands supported}}
  %1 = arith.ori %0, %0 : i32
}
