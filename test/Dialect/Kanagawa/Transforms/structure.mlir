// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s --kanagawa-call-prep | circt-opt | FileCheck %s --check-prefix=PREP


// CHECK-LABEL: kanagawa.class sym @C {
// CHECK:         kanagawa.method @getAndSet(%x: ui32) -> ui32 {
// CHECK:           kanagawa.return %x : ui32
// CHECK:         kanagawa.method @returnNothingWithRet() -> () {
// CHECK:           kanagawa.return

// PREP-LABEL: kanagawa.class sym @C {
// PREP:         kanagawa.method @getAndSet(%arg: !hw.struct<x: ui32>) -> ui32 {
// PREP:           %x = hw.struct_explode %arg : !hw.struct<x: ui32>
// PREP:           kanagawa.return %x : ui32
// PREP:         kanagawa.method @returnNothingWithRet(%arg: !hw.struct<>) -> () {
// PREP:           hw.struct_explode %arg : !hw.struct<>
// PREP:           kanagawa.return

kanagawa.design @foo {
kanagawa.class sym @C {
  kanagawa.method @getAndSet(%x: ui32) -> ui32 {
    kanagawa.return %x : ui32
  }
  kanagawa.method @returnNothingWithRet() {
    kanagawa.return
  }
}

// CHECK-LABEL: kanagawa.class sym @User {
// CHECK:         [[c:%.+]] = kanagawa.instance @c, <@foo::@C>
// CHECK:         kanagawa.method @getAndSetWrapper(%new_value: ui32) -> ui32 {
// CHECK:           [[x:%.+]] = kanagawa.call <@foo::@getAndSet>(%new_value) : (ui32) -> ui32
// CHECK:           kanagawa.return [[x]] : ui32
// CHECK:         kanagawa.method @getAndSetDup(%new_value: ui32) -> ui32 {
// CHECK:           [[x:%.+]] = kanagawa.call <@foo::@getAndSet>(%new_value) : (ui32) -> ui32
// CHECK:           kanagawa.return [[x]] : ui32


// PREP-LABEL: kanagawa.class sym @User {
// PREP:         [[c:%.+]] = kanagawa.instance @c, <@foo::@C>
// PREP:         kanagawa.method @getAndSetWrapper(%arg: !hw.struct<new_value: ui32>) -> ui32 {
// PREP:           %new_value = hw.struct_explode %arg : !hw.struct<new_value: ui32>
// PREP:           [[STRUCT1:%.+]] = hw.struct_create (%new_value) {sv.namehint = "getAndSet_args_called_from_getAndSetWrapper"} : !hw.struct<x: ui32>
// PREP:           [[CALLRES1:%.+]] = kanagawa.call <@foo::@getAndSet>([[STRUCT1]]) : (!hw.struct<x: ui32>) -> ui32
// PREP:         kanagawa.method @getAndSetDup(%arg: !hw.struct<new_value: ui32>) -> ui32 {
// PREP:           %new_value = hw.struct_explode %arg : !hw.struct<new_value: ui32>
// PREP:           [[STRUCT2:%.+]] = hw.struct_create (%new_value) {sv.namehint = "getAndSet_args_called_from_getAndSetDup"} : !hw.struct<x: ui32>
// PREP:           [[CALLRES2:%.+]] = kanagawa.call <@foo::@getAndSet>([[STRUCT2]]) : (!hw.struct<x: ui32>) -> ui32
// PREP:           kanagawa.return [[CALLRES2]] : ui32
kanagawa.class sym @User {
  kanagawa.instance @c, <@foo::@C>
  kanagawa.method @getAndSetWrapper(%new_value: ui32) -> ui32 {
    %x = kanagawa.call <@foo::@getAndSet>(%new_value): (ui32) -> ui32
    kanagawa.return %x : ui32
  }

  kanagawa.method @getAndSetDup(%new_value: ui32) -> ui32 {
    %x = kanagawa.call <@foo::@getAndSet>(%new_value): (ui32) -> ui32
    kanagawa.return %x : ui32
  }
}
}
