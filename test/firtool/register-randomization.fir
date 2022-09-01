; RUN: firtool -preserve-values=all -verilog %s | FileCheck %s --check-prefix=ALL
; RUN: firtool -preserve-values=named -verilog %s | FileCheck %s --check-prefix=NAMED
; RUN: firtool -preserve-values=none -verilog %s | FileCheck %s --check-prefix=NONE
; RUN: firtool -preserve-values=all -verilog %s --lowering-options=disallowLocalVariables| FileCheck %s --check-prefix=LOCAL

circuit Foo:
  module Foo:
    input clock: Clock
    input d: UInt<33>
    input d0: UInt<60000>
    input d1: UInt<120000>
    output q: UInt<33>
    output q0: UInt<60000>
    output q1: UInt<120000>


    ; ALL:       _r <= d;
    ; NAMED-NOT: _r = {{.*}};
    ; NONE-NOT:  _r = {{.*}};
    ; LOCAL:     _r <= d;
    reg _r: UInt<33>, clock
    _r <= d

    reg r: UInt<33>, clock
    r <= d

    reg s: UInt<33>, clock
    s <= d

    q <= s

    ; ALL:   automatic logic [31:0] _RANDOM_0;
    ; ALL:   automatic logic [31:0] _RANDOM_1;
    ; ALL:   automatic logic [31:0] _RANDOM_2;
    ; ALL:   automatic logic [31:0] _RANDOM_3;
    ; ALL:   _RANDOM_0 = `RANDOM;
    ; ALL:   _RANDOM_1 = `RANDOM;
    ; ALL:   _RANDOM_2 = `RANDOM;
    ; ALL:   _RANDOM_3 = `RANDOM;

    ; NAMED:   automatic logic [31:0] _RANDOM_0;
    ; NAMED:   automatic logic [31:0] _RANDOM_1;
    ; NAMED:   automatic logic [31:0] _RANDOM_2;
    ; NAMED:   automatic logic [31:0] _RANDOM_3;
    ; NAMED:   _RANDOM_0 = `RANDOM;
    ; NAMED:   _RANDOM_1 = `RANDOM;
    ; NAMED:   _RANDOM_2 = `RANDOM;
    ; NAMED:   _RANDOM_3 = `RANDOM;

    ; NONE:   automatic logic [31:0] _RANDOM_0;
    ; NONE:   automatic logic [31:0] _RANDOM_1;
    ; NONE:   automatic logic [31:0] _RANDOM_2;
    ; NONE:   automatic logic [31:0] _RANDOM_3;
    ; NONE:   _RANDOM_0 = `RANDOM;
    ; NONE:   _RANDOM_1 = `RANDOM;
    ; NONE:   _RANDOM_2 = `RANDOM;
    ; NONE:   _RANDOM_3 = `RANDOM;

    ; ALL:       _r = {_RANDOM_0, _RANDOM_1[0]}; 
    ; NAMED-NOT: _r = 
    ; NONE-NOT:  _r = 
    

    ; ALL:       r = {_RANDOM_1[31:1], _RANDOM_2[1:0]};
    ; NAMED:     r = {_RANDOM_1[31:1], _RANDOM_2[1:0]};
    ; NONE-NOT:  r = {{.*}};

    ; ALL:       s = {_RANDOM_2[31:2], _RANDOM_3[2:0]};
    ; NAMED:     s = {_RANDOM_2[31:2], _RANDOM_3[2:0]};
    ; NONE:      s = {_RANDOM_2[31:2], _RANDOM_3[2:0]};

    inst i0 of TwoLargeRegisters
    i0.clock <= clock
    i0.d <= d0
    q0 <= i0.q

    inst i1 of OneReallyLargeRegister
    i1.clock <= clock
    i1.d <= d1
    q1 <= i1.q

  ; ALL: module TwoLargeRegisters
  module TwoLargeRegisters:
    input clock: Clock
    input d: UInt<60000>
    output q: UInt<60000>
    ; ALL: automatic logic [31:0]   _RANDOM_0;
    ; ALL: automatic logic [31:0]   _RANDOM_3749;
    ; ALL-NOT: automatic logic [31:0]   _RANDOM_3750;

    reg r0: UInt<60000>, clock
    reg r1: UInt<60000>, clock

    r0 <= d
    r1 <= r0
    q <= r1

  ; ALL: module OneReallyLargeRegister
  module OneReallyLargeRegister:
    input clock: Clock
    input d: UInt<120000>
    output q: UInt<120000>

    ; ALL: automatic logic [31:0]   _RANDOM_0;
    ; ALL: automatic logic [31:0]   _RANDOM_3749;
    ; ALL-NOT: automatic logic [31:0]   _RANDOM_3750;

    reg r: UInt<120000>, clock

    r <= d
    q <= r
