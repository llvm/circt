// RUN: circt-opt %s | circt-opt | FileCheck %s

!exStruct1 = type !esi.struct<
    {exInt,   i1},
    {exFP,   f32}
>
!exStruct2 = type !esi.struct<{sint2, si2}, {float1, f32}>

!exUnion1 = type !esi.union<{struct1, !exStruct1}, {float1, f32}>

// CHECK-LABEL: module
module {
    // CHECK-LABEL: func @struct1(%{{.*}}: !esi.struct<{exInt,i1},{exFP,f32}>)
    func @struct1(%A: !exStruct1) {
        return
    }

    // CHECK-LABEL: func @union1(%arg0: !esi.struct<{struct1,!esi.struct<{exInt,i1},{exFP,f32}>},{float1,f32}>)
    func @union1(%A: !exUnion1) {
        return
    }

    // CHECK-LABEL:  func @enum1(%arg0: !esi.enum<["optionA", "doorNumB"]>)
    func @enum1(%A: !esi.enum<["optionA", "doorNumB"]>) {
        return
    }
}
