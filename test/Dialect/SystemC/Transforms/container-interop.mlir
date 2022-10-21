// RUN: circt-opt --systemc-lower-container-interop %s | FileCheck %s

emitc.include <"systemc.h">
emitc.include "VBar.h"
emitc.include "VEmpty.h"

systemc.module @Foo (%x: !systemc.in<!systemc.uint<32>>, %y: !systemc.out<!systemc.uint<32>>) {
  systemc.ctor {
    systemc.method %innerLogic
  }
  %innerLogic = systemc.func  {
    %0 = systemc.signal.read %x : !systemc.in<!systemc.uint<32>>
    %1 = systemc.convert %0 : (!systemc.uint<32>) -> i32
    %2 = interop.procedural.alloc cpp : !emitc.ptr<!emitc.opaque<"VBar">>
    interop.procedural.init cpp %2 : !emitc.ptr<!emitc.opaque<"VBar">> {
      %6 = systemc.cpp.new() : () -> !emitc.ptr<!emitc.opaque<"VBar">>
      interop.return %6 : !emitc.ptr<!emitc.opaque<"VBar">>
    }
    %3 = interop.procedural.update cpp [%2] (%1, %1) : [!emitc.ptr<!emitc.opaque<"VBar">>] (i32, i32) -> i32 {
    ^bb0(%arg0: !emitc.ptr<!emitc.opaque<"VBar">>, %arg1: i32, %arg2: i32):
      %6 = systemc.cpp.member_access %arg0 arrow "a" : (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
      systemc.cpp.assign %6 = %arg1 : i32
      %7 = systemc.cpp.member_access %arg0 arrow "b" : (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
      systemc.cpp.assign %7 = %arg2 : i32
      %8 = systemc.cpp.member_access %arg0 arrow "eval" : (!emitc.ptr<!emitc.opaque<"VBar">>) -> (() -> ())
      func.call_indirect %8() : () -> ()
      %9 = systemc.cpp.member_access %arg0 arrow "c" : (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
      interop.return %9 : i32
    }
    interop.procedural.dealloc cpp %2 : !emitc.ptr<!emitc.opaque<"VBar">> {
    ^bb0(%arg0: !emitc.ptr<!emitc.opaque<"VBar">>):
      systemc.cpp.delete %arg0 : !emitc.ptr<!emitc.opaque<"VBar">>
    }
    %4 = interop.procedural.alloc cpp : !emitc.ptr<!emitc.opaque<"VEmpty">>
    interop.procedural.init cpp %4 : !emitc.ptr<!emitc.opaque<"VEmpty">> {
      %6 = systemc.cpp.new() : () -> !emitc.ptr<!emitc.opaque<"VEmpty">>
      interop.return %6 : !emitc.ptr<!emitc.opaque<"VEmpty">>
    }
    interop.procedural.update cpp [%4] : [!emitc.ptr<!emitc.opaque<"VEmpty">>] () -> () {
    ^bb0(%arg0: !emitc.ptr<!emitc.opaque<"VEmpty">>):
      %6 = systemc.cpp.member_access %arg0 arrow "eval" : (!emitc.ptr<!emitc.opaque<"VEmpty">>) -> (() -> ())
      func.call_indirect %6() : () -> ()
      interop.return
    }
    interop.procedural.dealloc cpp %4 : !emitc.ptr<!emitc.opaque<"VEmpty">> {
    ^bb0(%arg0: !emitc.ptr<!emitc.opaque<"VEmpty">>):
      systemc.cpp.delete %arg0 : !emitc.ptr<!emitc.opaque<"VEmpty">>
    }
    %5 = systemc.convert %3 : (i32) -> !systemc.uint<32>
    systemc.signal.write %y, %5 : !systemc.out<!systemc.uint<32>>
  }
}