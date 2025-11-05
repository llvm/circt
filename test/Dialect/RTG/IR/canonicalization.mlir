// RUN: circt-opt --canonicalize %s | FileCheck %s

func.func @dummy(%arg0: !rtg.isa.label) -> () {return}
func.func @dummy1(%arg0: !rtg.string) -> () {return}
func.func @dummy2(%arg0: !rtg.array<index>) -> () {return}
func.func @dummy3(%arg0: !rtg.set<!rtg.tuple<index>>) -> () {return}
func.func @dummy4(%arg0: !rtg.set<index>) -> () {return}
func.func @dummy5(%arg0: !rtg.set<!rtg.tuple<index, i32, i64>>) -> () {return}
func.func @dummy6(%arg0: index) -> () {return}

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

// CHECK-LABEL: @sets
rtg.test @sets() {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %set0 = rtg.constant #rtg.set<1 : index, 0 : index> : !rtg.set<index> 
  %set1 = rtg.constant #rtg.set<1 : index, 2 : index> : !rtg.set<index> 
  %set2 = rtg.constant #rtg.set<2 : index, 3 : index> : !rtg.set<index> 
  %set3 = rtg.constant #rtg.set<> : !rtg.set<i64>
  %set4 = rtg.constant #rtg.set<4 : i32, 5 : i32> : !rtg.set<i32>
  %set5 = rtg.constant #rtg.set<6 : i64, 7 : i64> : !rtg.set<i64>

  // CHECK: [[SET:%.+]] = rtg.constant #rtg.set<0 : index, 1 : index> : !rtg.set<index>
  %0 = rtg.set_create %idx1, %idx0 : index

  // CHECK: [[SIZE:%.+]] = rtg.constant 2 : index
  %size = rtg.set_size %0 : !rtg.set<index>

  // CHECK: [[UNION:%.+]] = rtg.constant #rtg.set<0 : index, 1 : index, 2 : index, 3 : index> : !rtg.set<index>
  %union = rtg.set_union %set0, %set1, %set2 : !rtg.set<index>

  // CHECK: [[DIFF:%.+]] = rtg.constant #rtg.set<0 : index> : !rtg.set<index>
  %diff = rtg.set_difference %set0, %set1 : !rtg.set<index>

  // CHECK: [[PROD:%.+]] = rtg.constant #rtg.set<
  // CHECK-SAME: #rtg.tuple<0 : index, 4 : i32, 6 : i64> : !rtg.tuple<index, i32, i64>
  // CHECK-SAME: #rtg.tuple<0 : index, 4 : i32, 7 : i64> : !rtg.tuple<index, i32, i64>
  // CHECK-SAME: #rtg.tuple<0 : index, 5 : i32, 6 : i64> : !rtg.tuple<index, i32, i64>
  // CHECK-SAME: #rtg.tuple<0 : index, 5 : i32, 7 : i64> : !rtg.tuple<index, i32, i64>
  // CHECK-SAME: #rtg.tuple<1 : index, 4 : i32, 6 : i64> : !rtg.tuple<index, i32, i64>
  // CHECK-SAME: #rtg.tuple<1 : index, 4 : i32, 7 : i64> : !rtg.tuple<index, i32, i64>
  // CHECK-SAME: #rtg.tuple<1 : index, 5 : i32, 6 : i64> : !rtg.tuple<index, i32, i64>
  // CHECK-SAME: #rtg.tuple<1 : index, 5 : i32, 7 : i64> : !rtg.tuple<index, i32, i64>>
  // CHECK-SAME: !rtg.set<!rtg.tuple<index, i32, i64>>
  %prod0 = rtg.set_cartesian_product %set0, %set4, %set5 : !rtg.set<index>, !rtg.set<i32>, !rtg.set<i64>
  // CHECK: [[EMPTY:%.+]] = rtg.constant #rtg.set<> : !rtg.set<!rtg.tuple<index, i32, i64>>
  %prod1 = rtg.set_cartesian_product %set0, %set4, %set3 : !rtg.set<index>, !rtg.set<i32>, !rtg.set<i64>
  // CHECK: [[SET2:%.+]] = rtg.constant #rtg.set<#rtg.tuple<0 : index> : !rtg.tuple<index>, #rtg.tuple<1 : index> : !rtg.tuple<index>> : !rtg.set<!rtg.tuple<index>>
  %prod2 = rtg.set_cartesian_product %set0 : !rtg.set<index>

  // CHECK: func.call @dummy4([[SET:%.+]])
  func.call @dummy4(%0) : (!rtg.set<index>) -> ()
  // CHECK: func.call @dummy6([[SIZE:%.+]])
  func.call @dummy6(%size) : (index) -> ()
  // CHECK: func.call @dummy4([[UNION]])
  func.call @dummy4(%union) : (!rtg.set<index>) -> ()
  // CHECK: func.call @dummy4([[DIFF]])
  func.call @dummy4(%diff) : (!rtg.set<index>) -> ()
  // CHECK: func.call @dummy5([[PROD]])
  func.call @dummy5(%prod0) : (!rtg.set<!rtg.tuple<index, i32, i64>>) -> ()
  // CHECK: func.call @dummy5([[EMPTY]])
  func.call @dummy5(%prod1) : (!rtg.set<!rtg.tuple<index, i32, i64>>) -> ()
  // CHECK: func.call @dummy3([[SET2]]) : (!rtg.set<!rtg.tuple<index>>) -> ()
  func.call @dummy3(%prod2) : (!rtg.set<!rtg.tuple<index>>) -> ()
}
