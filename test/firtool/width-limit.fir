; RUN: not firtool %s --format=fir --split-input-file 2>&1 | FileCheck %s

; CHECK-DAG: error: integer width 1040498778 exceeds the maximum supported width of 16777215
; CHECK-DAG: error: integer width 16777216 exceeds the maximum supported width of 16777215
; CHECK-DAG: module Max(
; CHECK-DAG: input {{.*}}[16777214:0] a
; CHECK-DAG: output {{.*}}[16777214:0] b
; CHECK-DAG: module Small(
; CHECK-DAG: input {{.*}}[7:0] a
; CHECK-DAG: output {{.*}}[7:0] b
; CHECK-DAG: input {{.*}}[31:0] c
; CHECK-DAG: output {{.*}}[31:0] d

FIRRTL version 7.0.0
circuit Exceeds :
  public module Exceeds :
    input a : UInt<1040498778>
    output b : UInt<1040498778>
    connect b, a

; // -----

FIRRTL version 7.0.0
circuit Over :
  public module Over :
    input a : UInt<16777216>
    output b : UInt<16777216>
    connect b, a

; // -----

FIRRTL version 7.0.0
circuit Max :
  public module Max :
    input a : UInt<16777215>
    output b : UInt<16777215>
    connect b, a

; // -----

FIRRTL version 7.0.0
circuit Small :
  public module Small :
    input a : UInt<8>
    output b : UInt<8>
    input c : SInt<32>
    output d : SInt<32>
    connect b, a
    connect d, c
