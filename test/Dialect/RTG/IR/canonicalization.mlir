// RUN: circt-opt --canonicalize %s | FileCheck %s

func.func @dummy(%arg0: !rtg.isa.label) -> () {return}
func.func @dummy1(%arg0: !rtg.string) -> () {return}
func.func @dummy2(%arg0: !rtg.array<index>) -> () {return}
func.func @dummy6(%arg0: index) -> () {return}
func.func @dummy7(%arg0: !rtgtest.ireg) -> () {return}

// CHECK-LABEL: @interleaveSequences
rtg.test @interleaveSequences(seq0 = %seq0: !rtg.randomized_sequence) {
  // CHECK-NEXT: rtg.embed_sequence %seq0
  %0 = rtg.interleave_sequences %seq0
  rtg.embed_sequence %0
}

// CHECK-LABEL: @immediates
rtg.target @immediates : !rtg.dict<imm0: !rtg.isa.immediate<64>, imm1: !rtg.isa.immediate<2>> {
  %0 = rtg.constant #rtg.isa.immediate<32, -1>
  %1 = rtg.constant #rtg.isa.immediate<32, 0>
  %3 = rtg.isa.concat_immediate %1, %0 : !rtg.isa.immediate<32>, !rtg.isa.immediate<32>
  %4 = rtg.isa.slice_immediate %3 from 31 : !rtg.isa.immediate<64> -> !rtg.isa.immediate<2>
  
  // CHECK-NEXT: [[V0:%.+]] = rtg.constant #rtg.isa.immediate<64, 4294967295>
  // CHECK-NEXT: [[V1:%.+]] = rtg.constant #rtg.isa.immediate<2, 1>
  // CHECK-NEXT: rtg.yield [[V0]], [[V1]] :
  rtg.yield %3, %4 : !rtg.isa.immediate<64>, !rtg.isa.immediate<2>
}

// CHECK-LABEL: @labels
rtg.test @labels() {
  %str = rtg.constant "label" : !rtg.string
  %0 = rtg.string_to_label %str 
  // CHECK: [[V0:%.+]] = rtg.constant #rtg.isa.label<"label">
  // CHECK: func.call @dummy([[V0]])
  func.call @dummy(%0) : (!rtg.isa.label) -> ()
}

// CHECK-LABEL: @constraints
rtg.test @constraints() {
  // CHECK: [[V0:%.+]] = rtg.constant false
  // CHECK: rtg.constraint [[V0]]
  %false = rtg.constant false
  rtg.constraint %false
  // CHECK-NOT: rtg.constraint
  %true = rtg.constant true
  rtg.constraint %true
}

// CHECK-LABEL: rtg.test @strings
rtg.test @strings() {
  // CHECK-DAG: [[V2:%.+]] = rtg.constant "" : !rtg.string
  // CHECK-DAG: [[V1:%.+]] = rtg.constant "23" : !rtg.string
  // CHECK-DAG: [[V0:%.+]] = rtg.constant "hellohello" : !rtg.string
  // CHECK-DAG: [[V3:%.+]] = rtg.constant "0x2A" : !rtg.string
  // CHECK-DAG: [[V4:%.+]] = rtg.constant "t0" : !rtg.string

  %0 = rtg.constant "hello" : !rtg.string
  %1 = rtg.string_concat %0, %0
  // CHECK-NEXT: func.call @dummy1([[V0]])
  func.call @dummy1(%1) : (!rtg.string) -> ()

  %2 = rtg.constant 23 : index
  %3 = rtg.int_format %2
  // CHECK-NEXT: func.call @dummy1([[V1]])
  func.call @dummy1(%3) : (!rtg.string) -> ()

  %4 = rtg.string_concat
  // CHECK-NEXT: func.call @dummy1([[V2]])
  func.call @dummy1(%4) : (!rtg.string) -> ()

  %imm = rtg.constant #rtg.isa.immediate<8, 42>
  %5 = rtg.immediate_format %imm : !rtg.isa.immediate<8>
  // CHECK-NEXT: func.call @dummy1([[V3]])
  func.call @dummy1(%5) : (!rtg.string) -> ()

  %reg = rtg.constant #rtgtest.t0 : !rtgtest.ireg
  %6 = rtg.register_format %reg : !rtgtest.ireg
  // CHECK-NEXT: func.call @dummy1([[V4]])
  func.call @dummy1(%6) : (!rtg.string) -> ()
}

// CHECK-LABEL: @arrays
rtg.test @arrays() {
  // CHECK-NEXT: [[IDX:%.+]] = index.constant 1
  %idx1 = index.constant 1
  %1 = rtg.array_create : index
  %2 = rtg.array_append %1, %idx1 : !rtg.array<index>
  // CHECK-NEXT: [[V1:%.+]] = rtg.array_create [[IDX]] : index
  // CHECK-NEXT: func.call @dummy2([[V1]])
  func.call @dummy2(%2) : (!rtg.array<index>) -> ()

  // CHECK-NEXT: [[V2:%.+]] = rtg.array_create [[IDX]], [[IDX]], [[IDX]] : index
  // CHECK-NEXT: func.call @dummy2([[V2]])
  %3 = rtg.array_create %idx1, %idx1 : index
  %4 = rtg.array_append %3, %idx1 : !rtg.array<index>
  func.call @dummy2(%4) : (!rtg.array<index>) -> ()
}

// CHECK-LABEL: @testRegisterToIndex
rtg.test @testRegisterToIndex() {
  %0 = rtg.constant #rtgtest.t0 : !rtgtest.ireg
  // CHECK-NEXT: [[V0:%.+]] = rtg.constant 5 : index
  // CHECK-NEXT: func.call @dummy6([[V0]])
  %1 = rtg.isa.register_to_index %0 : !rtgtest.ireg
  func.call @dummy6(%1) : (index) -> ()
}

// CHECK-LABEL: @testIndexToRegister
rtg.test @testIndexToRegister() {
  // CHECK-NEXT: [[T6:%.+]] = rtg.constant #rtgtest.t6 : !rtgtest.ireg
  // CHECK-NEXT: [[T5:%.+]] = rtg.constant #rtgtest.t5 : !rtgtest.ireg
  // CHECK-NEXT: [[T4:%.+]] = rtg.constant #rtgtest.t4 : !rtgtest.ireg
  // CHECK-NEXT: [[T3:%.+]] = rtg.constant #rtgtest.t3 : !rtgtest.ireg
  // CHECK-NEXT: [[S11:%.+]] = rtg.constant #rtgtest.s11 : !rtgtest.ireg
  // CHECK-NEXT: [[S10:%.+]] = rtg.constant #rtgtest.s10 : !rtgtest.ireg
  // CHECK-NEXT: [[S9:%.+]] = rtg.constant #rtgtest.s9 : !rtgtest.ireg
  // CHECK-NEXT: [[S8:%.+]] = rtg.constant #rtgtest.s8 : !rtgtest.ireg
  // CHECK-NEXT: [[S7:%.+]] = rtg.constant #rtgtest.s7 : !rtgtest.ireg
  // CHECK-NEXT: [[S6:%.+]] = rtg.constant #rtgtest.s6 : !rtgtest.ireg
  // CHECK-NEXT: [[S5:%.+]] = rtg.constant #rtgtest.s5 : !rtgtest.ireg
  // CHECK-NEXT: [[S4:%.+]] = rtg.constant #rtgtest.s4 : !rtgtest.ireg
  // CHECK-NEXT: [[S3:%.+]] = rtg.constant #rtgtest.s3 : !rtgtest.ireg
  // CHECK-NEXT: [[S2:%.+]] = rtg.constant #rtgtest.s2 : !rtgtest.ireg
  // CHECK-NEXT: [[A7:%.+]] = rtg.constant #rtgtest.a7 : !rtgtest.ireg
  // CHECK-NEXT: [[A6:%.+]] = rtg.constant #rtgtest.a6 : !rtgtest.ireg
  // CHECK-NEXT: [[A5:%.+]] = rtg.constant #rtgtest.a5 : !rtgtest.ireg
  // CHECK-NEXT: [[A4:%.+]] = rtg.constant #rtgtest.a4 : !rtgtest.ireg
  // CHECK-NEXT: [[A3:%.+]] = rtg.constant #rtgtest.a3 : !rtgtest.ireg
  // CHECK-NEXT: [[A2:%.+]] = rtg.constant #rtgtest.a2 : !rtgtest.ireg
  // CHECK-NEXT: [[A1:%.+]] = rtg.constant #rtgtest.a1 : !rtgtest.ireg
  // CHECK-NEXT: [[A0:%.+]] = rtg.constant #rtgtest.a0 : !rtgtest.ireg
  // CHECK-NEXT: [[S1:%.+]] = rtg.constant #rtgtest.s1 : !rtgtest.ireg
  // CHECK-NEXT: [[S0:%.+]] = rtg.constant #rtgtest.s0 : !rtgtest.ireg
  // CHECK-NEXT: [[T2:%.+]] = rtg.constant #rtgtest.t2 : !rtgtest.ireg
  // CHECK-NEXT: [[T1:%.+]] = rtg.constant #rtgtest.t1 : !rtgtest.ireg
  // CHECK-NEXT: [[T0:%.+]] = rtg.constant #rtgtest.t0 : !rtgtest.ireg
  // CHECK-NEXT: [[TP:%.+]] = rtg.constant #rtgtest.tp : !rtgtest.ireg
  // CHECK-NEXT: [[GP:%.+]] = rtg.constant #rtgtest.gp : !rtgtest.ireg
  // CHECK-NEXT: [[SP:%.+]] = rtg.constant #rtgtest.sp : !rtgtest.ireg
  // CHECK-NEXT: [[RA:%.+]] = rtg.constant #rtgtest.ra : !rtgtest.ireg
  // CHECK-NEXT: [[ZERO:%.+]] = rtg.constant #rtgtest.zero : !rtgtest.ireg
  // CHECK-NEXT: func.call @dummy7([[ZERO]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[RA]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[SP]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[GP]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[TP]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[T0]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[T1]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[T2]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[S0]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[S1]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[A0]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[A1]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[A2]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[A3]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[A4]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[A5]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[A6]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[A7]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[S2]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[S3]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[S4]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[S5]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[S6]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[S7]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[S8]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[S9]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[S10]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[S11]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[T3]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[T4]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[T5]]) : (!rtgtest.ireg) -> ()
  // CHECK-NEXT: func.call @dummy7([[T6]]) : (!rtgtest.ireg) -> ()

  %c0 = index.constant 0
  %c1 = index.constant 1
  %c2 = index.constant 2
  %c3 = index.constant 3
  %c4 = index.constant 4
  %c5 = index.constant 5
  %c6 = index.constant 6
  %c7 = index.constant 7
  %c8 = index.constant 8
  %c9 = index.constant 9
  %c10 = index.constant 10
  %c11 = index.constant 11
  %c12 = index.constant 12
  %c13 = index.constant 13
  %c14 = index.constant 14
  %c15 = index.constant 15
  %c16 = index.constant 16
  %c17 = index.constant 17
  %c18 = index.constant 18
  %c19 = index.constant 19
  %c20 = index.constant 20
  %c21 = index.constant 21
  %c22 = index.constant 22
  %c23 = index.constant 23
  %c24 = index.constant 24
  %c25 = index.constant 25
  %c26 = index.constant 26
  %c27 = index.constant 27
  %c28 = index.constant 28
  %c29 = index.constant 29
  %c30 = index.constant 30
  %c31 = index.constant 31
  %0 = rtg.isa.index_to_register %c0 : !rtgtest.ireg
  func.call @dummy7(%0) : (!rtgtest.ireg) -> ()
  %1 = rtg.isa.index_to_register %c1 : !rtgtest.ireg
  func.call @dummy7(%1) : (!rtgtest.ireg) -> ()
  %2 = rtg.isa.index_to_register %c2 : !rtgtest.ireg
  func.call @dummy7(%2) : (!rtgtest.ireg) -> ()
  %3 = rtg.isa.index_to_register %c3 : !rtgtest.ireg
  func.call @dummy7(%3) : (!rtgtest.ireg) -> ()
  %4 = rtg.isa.index_to_register %c4 : !rtgtest.ireg
  func.call @dummy7(%4) : (!rtgtest.ireg) -> ()
  %5 = rtg.isa.index_to_register %c5 : !rtgtest.ireg
  func.call @dummy7(%5) : (!rtgtest.ireg) -> ()
  %6 = rtg.isa.index_to_register %c6 : !rtgtest.ireg
  func.call @dummy7(%6) : (!rtgtest.ireg) -> ()
  %7 = rtg.isa.index_to_register %c7 : !rtgtest.ireg
  func.call @dummy7(%7) : (!rtgtest.ireg) -> ()
  %8 = rtg.isa.index_to_register %c8 : !rtgtest.ireg
  func.call @dummy7(%8) : (!rtgtest.ireg) -> ()
  %9 = rtg.isa.index_to_register %c9 : !rtgtest.ireg
  func.call @dummy7(%9) : (!rtgtest.ireg) -> ()
  %10 = rtg.isa.index_to_register %c10 : !rtgtest.ireg
  func.call @dummy7(%10) : (!rtgtest.ireg) -> ()
  %11 = rtg.isa.index_to_register %c11 : !rtgtest.ireg
  func.call @dummy7(%11) : (!rtgtest.ireg) -> ()
  %12 = rtg.isa.index_to_register %c12 : !rtgtest.ireg
  func.call @dummy7(%12) : (!rtgtest.ireg) -> ()
  %13 = rtg.isa.index_to_register %c13 : !rtgtest.ireg
  func.call @dummy7(%13) : (!rtgtest.ireg) -> ()
  %14 = rtg.isa.index_to_register %c14 : !rtgtest.ireg
  func.call @dummy7(%14) : (!rtgtest.ireg) -> ()
  %15 = rtg.isa.index_to_register %c15 : !rtgtest.ireg
  func.call @dummy7(%15) : (!rtgtest.ireg) -> ()
  %16 = rtg.isa.index_to_register %c16 : !rtgtest.ireg
  func.call @dummy7(%16) : (!rtgtest.ireg) -> ()
  %17 = rtg.isa.index_to_register %c17 : !rtgtest.ireg
  func.call @dummy7(%17) : (!rtgtest.ireg) -> ()
  %18 = rtg.isa.index_to_register %c18 : !rtgtest.ireg
  func.call @dummy7(%18) : (!rtgtest.ireg) -> ()
  %19 = rtg.isa.index_to_register %c19 : !rtgtest.ireg
  func.call @dummy7(%19) : (!rtgtest.ireg) -> ()
  %20 = rtg.isa.index_to_register %c20 : !rtgtest.ireg
  func.call @dummy7(%20) : (!rtgtest.ireg) -> ()
  %21 = rtg.isa.index_to_register %c21 : !rtgtest.ireg
  func.call @dummy7(%21) : (!rtgtest.ireg) -> ()
  %22 = rtg.isa.index_to_register %c22 : !rtgtest.ireg
  func.call @dummy7(%22) : (!rtgtest.ireg) -> ()
  %23 = rtg.isa.index_to_register %c23 : !rtgtest.ireg
  func.call @dummy7(%23) : (!rtgtest.ireg) -> ()
  %24 = rtg.isa.index_to_register %c24 : !rtgtest.ireg
  func.call @dummy7(%24) : (!rtgtest.ireg) -> ()
  %25 = rtg.isa.index_to_register %c25 : !rtgtest.ireg
  func.call @dummy7(%25) : (!rtgtest.ireg) -> ()
  %26 = rtg.isa.index_to_register %c26 : !rtgtest.ireg
  func.call @dummy7(%26) : (!rtgtest.ireg) -> ()
  %27 = rtg.isa.index_to_register %c27 : !rtgtest.ireg
  func.call @dummy7(%27) : (!rtgtest.ireg) -> ()
  %28 = rtg.isa.index_to_register %c28 : !rtgtest.ireg
  func.call @dummy7(%28) : (!rtgtest.ireg) -> ()
  %29 = rtg.isa.index_to_register %c29 : !rtgtest.ireg
  func.call @dummy7(%29) : (!rtgtest.ireg) -> ()
  %30 = rtg.isa.index_to_register %c30 : !rtgtest.ireg
  func.call @dummy7(%30) : (!rtgtest.ireg) -> ()
  %31 = rtg.isa.index_to_register %c31 : !rtgtest.ireg
  func.call @dummy7(%31) : (!rtgtest.ireg) -> ()
}
