// RUN: circt-opt --canonicalize %s | FileCheck %s

func.func @dummy(%arg0: !rtg.isa.label) -> () {return}
func.func @dummy1(%arg0: !rtg.string) -> () {return}
func.func @dummy2(%arg0: !rtg.array<index>) -> () {return}

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
