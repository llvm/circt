// RUN: circt-opt %s --arc-lower-taps | FileCheck %s

arc.model "Basic" {
^bb0(%arg0: !arc.storage<12>):
  %0 = arc.storage.get %arg0[0] : !arc.storage<12> -> !arc.state<i32>
  arc.state_tap %0 input rw "root/input" : !arc.state<i32>

// CHECK:      systemc.cpp.func @get_root_input(%state: !emitc.ptr<!emitc.opaque<"void">>) -> i32 {
// CHECK-NEXT:   %0 = "emitc.constant"() <{value = #emitc.opaque<"((uint8_t*) state + 0)">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
// CHECK-NEXT:   %1 = emitc.cast %0 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// CHECK-NEXT:   %2 = emitc.apply "*"(%1) : (!emitc.ptr<i32>) -> i32
// CHECK-NEXT:   systemc.cpp.return %2 : i32
// CHECK-NEXT: }
// CHECK-NEXT: systemc.cpp.func @set_root_input(%state: !emitc.ptr<!emitc.opaque<"void">>, %value: i32) {
// CHECK-NEXT:   %0 = "emitc.constant"() <{value = #emitc.opaque<"((uint8_t*) state + 0)">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
// CHECK-NEXT:   %1 = emitc.cast %0 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// CHECK-NEXT:   %2 = emitc.apply "*"(%1) : (!emitc.ptr<i32>) -> i32
// CHECK-NEXT:   systemc.cpp.assign %2 = %value : i32
// CHECK-NEXT:   systemc.cpp.return
// CHECK-NEXT: }

  %1 = arc.storage.get %arg0[4] : !arc.storage<12> -> !arc.memory<2 x i32, i1>
  arc.state_tap %1 memory rw "intern/mem" : !arc.memory<2 x i32, i1>

// CHECK-NEXT: systemc.cpp.func @get_intern_mem(%state: !emitc.ptr<!emitc.opaque<"void">>, %idx: i1) -> i32 {
// CHECK-NEXT:   %0 = "emitc.constant"() <{value = #emitc.opaque<"((uint8_t*) state + 4 + 4 * idx)">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
// CHECK-NEXT:   %1 = emitc.cast %0 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// CHECK-NEXT:   %2 = emitc.apply "*"(%1) : (!emitc.ptr<i32>) -> i32
// CHECK-NEXT:   systemc.cpp.return %2 : i32
// CHECK-NEXT: }
// CHECK-NEXT: systemc.cpp.func @set_intern_mem(%state: !emitc.ptr<!emitc.opaque<"void">>, %value: i32, %idx: i1) {
// CHECK-NEXT:   %0 = "emitc.constant"() <{value = #emitc.opaque<"((uint8_t*) state + 4 + 4 * idx)">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
// CHECK-NEXT:   %1 = emitc.cast %0 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// CHECK-NEXT:   %2 = emitc.apply "*"(%1) : (!emitc.ptr<i32>) -> i32
// CHECK-NEXT:   systemc.cpp.assign %2 = %value : i32
// CHECK-NEXT:   systemc.cpp.return
// CHECK-NEXT: }
}

// CHECK-NEXT: systemc.cpp.func @alloc_and_init_Basic() -> !emitc.ptr<!emitc.opaque<"void">> {
// CHECK-NEXT:   %0 = emitc.call "calloc"() {args = [#emitc.opaque<"1">, #emitc.opaque<"12">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
// CHECK-NEXT:   systemc.cpp.return %0 : !emitc.ptr<!emitc.opaque<"void">>
// CHECK-NEXT: }
