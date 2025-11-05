// RUN: circt-opt --canonicalize %s | FileCheck %s

func.func @dummy0(%arg0: !rtg.set<index>) -> () {return}
func.func @dummy1(%arg0: index) -> () {return}
func.func @dummy2(%arg0: !rtg.set<!rtg.tuple<index, i32, i64>>) -> () {return}
func.func @dummy3(%arg0: !rtg.set<!rtg.tuple<index>>) -> () {return}

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

  // CHECK: func.call @dummy0([[SET:%.+]])
  func.call @dummy0(%0) : (!rtg.set<index>) -> ()
  // CHECK: func.call @dummy1([[SIZE:%.+]])
  func.call @dummy1(%size) : (index) -> ()
  // CHECK: func.call @dummy0([[UNION]])
  func.call @dummy0(%union) : (!rtg.set<index>) -> ()
  // CHECK: func.call @dummy0([[DIFF]])
  func.call @dummy0(%diff) : (!rtg.set<index>) -> ()
  // CHECK: func.call @dummy2([[PROD]])
  func.call @dummy2(%prod0) : (!rtg.set<!rtg.tuple<index, i32, i64>>) -> ()
  // CHECK: func.call @dummy2([[EMPTY]])
  func.call @dummy2(%prod1) : (!rtg.set<!rtg.tuple<index, i32, i64>>) -> ()
  // CHECK: func.call @dummy3([[SET2]]) : (!rtg.set<!rtg.tuple<index>>) -> ()
  func.call @dummy3(%prod2) : (!rtg.set<!rtg.tuple<index>>) -> ()
}
