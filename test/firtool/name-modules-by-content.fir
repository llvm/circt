; RUN: firtool --split-input-file --name-modules-by-content %s | FileCheck %s
; RUN: firtool --split-input-file %s | FileCheck %s --check-prefix=OFF

; Two versions of one design. The second adds a 16-bit queue that Chisel
; elaborates first, so every queue after it is numbered one higher. With
; --name-modules-by-content the 8-bit and the 4-bit queue keep their names.

FIRRTL version 4.0.0
circuit Top:
  ; CHECK: module [[$WIDE:Queue_[0-9a-f]{8}]](
  ; OFF: module Queue(
  module Queue:
    input in : UInt<8>
    output out : UInt<8>
    connect out, in

  ; CHECK: module [[$NARROW:Queue_[0-9a-f]{8}]](
  ; OFF: module Queue_1(
  module Queue_1:
    input in : UInt<4>
    output out : UInt<4>
    connect out, in

  ; CHECK: module Top(
  ; CHECK: [[$WIDE]] q8 (
  ; CHECK: [[$NARROW]] q4 (
  public module Top:
    input a : UInt<8>
    input b : UInt<4>
    output x : UInt<8>
    output y : UInt<4>
    inst q8 of Queue
    inst q4 of Queue_1
    connect q8.in, a
    connect q4.in, b
    connect x, q8.out
    connect y, q4.out

; // -----

FIRRTL version 4.0.0
circuit Top:
  ; CHECK: module {{Queue_[0-9a-f]{8}[(]}}
  ; OFF: module Queue(
  module Queue:
    input in : UInt<16>
    output out : UInt<16>
    connect out, in

  ; CHECK: module [[$WIDE]](
  ; OFF: module Queue_1(
  module Queue_1:
    input in : UInt<8>
    output out : UInt<8>
    connect out, in

  ; CHECK: module [[$NARROW]](
  ; OFF: module Queue_2(
  module Queue_2:
    input in : UInt<4>
    output out : UInt<4>
    connect out, in

  public module Top:
    input a : UInt<8>
    input b : UInt<4>
    input c : UInt<16>
    output x : UInt<8>
    output y : UInt<4>
    output z : UInt<16>
    inst q16 of Queue
    inst q8 of Queue_1
    inst q4 of Queue_2
    connect q16.in, c
    connect q8.in, a
    connect q4.in, b
    connect x, q8.out
    connect y, q4.out
    connect z, q16.out
