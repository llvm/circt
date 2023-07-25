// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: ibis.class @C {
// CHECK:         ibis.method @getAndSet(%x: ui32) -> ui32 {
// CHECK:           ibis.return %x : ui32
// CHECK:         ibis.method @returnNothing() {
// CHECK:           ibis.return
// CHECK:         ibis.method @returnNothingWithRet() {
// CHECK:           ibis.return
ibis.class @C {
  ibis.method @getAndSet(%x: ui32) -> ui32 {
    ibis.return %x : ui32
  }
  ibis.method @returnNothing() {}
  ibis.method @returnNothingWithRet() {
    ibis.return
  }
}

// CHECK-LABEL: ibis.class @User {
// CHECK:         ibis.instance @c, @C
// CHECK:         ibis.method @getAndSetWrapper(%new_value: ui32) -> ui32 {
// CHECK:           [[x:%.+]] = ibis.call @c::@getAndSet(%new_value) : (ui32) -> ui32
// CHECK:           ibis.return [[x]] : ui32
// CHECK:         ibis.method @getAndSetDup(%new_value: ui32) -> ui32 {
// CHECK:           [[x:%.+]] = ibis.call @c::@getAndSet(%new_value) : (ui32) -> ui32
// CHECK:           ibis.return [[x]] : ui32
ibis.class @User {
  ibis.instance @c, @C
  ibis.method @getAndSetWrapper(%new_value: ui32) -> ui32 {
    %x = ibis.call @c::@getAndSet(%new_value): (ui32) -> ui32
    ibis.return %x : ui32
  }

  ibis.method @getAndSetDup(%new_value: ui32) -> ui32 {
    %x = ibis.call @c::@getAndSet(%new_value): (ui32) -> ui32
    ibis.return %x : ui32
  }
}
