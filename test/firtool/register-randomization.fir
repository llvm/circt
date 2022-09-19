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

    q <= r

    ; ALL:        automatic logic [31:0] _RANDOM_0;
    ; ALL-NEXT:   automatic logic [31:0] _RANDOM_1;
    ; ALL-NEXT:   automatic logic [31:0] _RANDOM_2;
    ; ALL-NEXT:   automatic logic [31:0] _RANDOM_3;
    ; ALL:        _RANDOM_0 = `RANDOM;
    ; ALL-NEXT:   _RANDOM_1 = `RANDOM;
    ; ALL-NEXT:   _RANDOM_2 = `RANDOM;
    ; ALL-NEXT:   _RANDOM_3 = `RANDOM;

    ; NAMED:        automatic logic [31:0] _RANDOM_0;
    ; NAMED-NEXT:   automatic logic [31:0] _RANDOM_1;
    ; NAMED-NEXT:   automatic logic [31:0] _RANDOM_2;
    ; NAMED-NEXT:   automatic logic [31:0] _RANDOM_3;
    ; NAMED:        _RANDOM_0 = `RANDOM;
    ; NAMED-NEXT:   _RANDOM_1 = `RANDOM;
    ; NAMED-NEXT:   _RANDOM_2 = `RANDOM;
    ; NAMED-NEXT:   _RANDOM_3 = `RANDOM;

    ; NONE:        automatic logic [31:0] _RANDOM_0;
    ; NONE-NEXT:   automatic logic [31:0] _RANDOM_1;
    ; NONE-NEXT:   automatic logic [31:0] _RANDOM_2;
    ; NONE:        _RANDOM_0 = `RANDOM;
    ; NONE-NEXT:   _RANDOM_1 = `RANDOM;
    ; NONE-NEXT:   _RANDOM_2 = `RANDOM;

    ; ALL:       _r = {_RANDOM_0, _RANDOM_1[0]};
    ; NAMED-NOT: _r =
    ; NONE-NOT:  _r =

    ; ALL:       r = {_RANDOM_1[31:1], _RANDOM_2[1:0]};
    ; NAMED:     r = {_RANDOM_1[31:1], _RANDOM_2[1:0]};
    ; NONE:      r = {_RANDOM_1[31:1], _RANDOM_2[1:0]};

    ; ALL:       s = {_RANDOM_2[31:2], _RANDOM_3[2:0]};
    ; NAMED:     s = {_RANDOM_2[31:2], _RANDOM_3[2:0]};
    ; NONE-NOT:  s = {{.*}};
