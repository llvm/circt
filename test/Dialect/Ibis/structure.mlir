// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: ibis.class @C {
// CHECK:         ibis.func @getAndSet(%x: ui32) -> ui32 {
// CHECK:           ibis.return %x : ui32
// CHECK:         ibis.func @returnNothing() {
// CHECK:           ibis.return
// CHECK:         ibis.func @returnNothingWithRet() {
// CHECK:           ibis.return
ibis.class @C {
  ibis.func @getAndSet(%x: ui32) -> ui32 {
    ibis.return %x : ui32
  }
  ibis.func @returnNothing() {}
  ibis.func @returnNothingWithRet() {
    ibis.return
  }
}

// CHECK-LABEL: ibis.class @User {
// CHECK:         ibis.instance @c, @C
// CHECK:         ibis.func @getAndSetWrapper(%new_value: ui32) -> ui32 {
// CHECK:           [[x:%.+]] = ibis.call @c::@getAndSet(%new_value) : (ui32) -> ui32
// CHECK:           ibis.return [[x]] : ui32
// CHECK:         ibis.func @getAndSetDup(%new_value: ui32) -> ui32 {
// CHECK:           [[x:%.+]] = ibis.call @c::@getAndSet(%new_value) : (ui32) -> ui32
// CHECK:           ibis.return [[x]] : ui32
ibis.class @User {
  ibis.instance @c, @C
  ibis.func @getAndSetWrapper(%new_value: ui32) -> ui32 {
    %x = ibis.call @c::@getAndSet(%new_value): (ui32) -> ui32
    ibis.return %x : ui32
  }

  ibis.func @getAndSetDup(%new_value: ui32) -> ui32 {
    %x = ibis.call @c::@getAndSet(%new_value): (ui32) -> ui32
    ibis.return %x : ui32
  }
}
