// RUN: circt-opt %s | circt-opt | FileCheck %s

!exStruct1 = type !rtl.struct<
  {exInt,   i1},
  {exFP,   f32}
>
!exStruct2 = type !rtl.struct<{sint2, si2}, {float1, f32}>

!exUnion1 = type !rtl.union<{struct1, !exStruct1}, {float1, f32}>

// CHECK-LABEL: module
module {
  // CHECK-LABEL: func @struct1(%{{.*}}: !rtl.struct<{exInt,i1},{exFP,f32}>)
  func @struct1(%A: !exStruct1) {
    return
  }

  // CHECK-LABEL: func @union1(%arg0: !rtl.union<{struct1,!rtl.struct<{exInt,i1},{exFP,f32}>},{float1,f32}>)
  func @union1(%A: !exUnion1) {
    return
  }

  // CHECK-LABEL:  func @enum1(%arg0: !rtl.enum<optionA,doorNumB>)
  func @enum1(%A: !rtl.enum<optionA, doorNumB>) {
    return
  }

  // CHECK-LABEL: func @struct(%{{.*}}: !rtl.struct<{i,i1},{f,tensor<4x2xi4>}>)
  func @struct(%A: !rtl.struct<{i, i1}, {f, tensor<4 x 2 x i4> }>) {
    return
  }
}
