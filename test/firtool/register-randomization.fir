; RUN: firtool -preserve-values=all -verilog %s | FileCheck %s --check-prefix=ALL
; RUN: firtool -preserve-values=named -verilog %s | FileCheck %s --check-prefix=NAMED
; RUN: firtool -preserve-values=none -verilog %s | FileCheck %s --check-prefix=NONE

circuit Foo:
  module Foo:
    input clock: Clock
    input d: UInt<33>
    input d0: UInt<60000>
    input d1: UInt<120000>
    output q: UInt<33>
    output q0: UInt<60000>
    output q1: UInt<120000>

    ; ALL:   reg [127:0] _RANDOM;
    ; ALL:   _RANDOM = {`RANDOM,`RANDOM,`RANDOM,`RANDOM};

    ; NAMED: reg [127:0] _RANDOM;
    ; NAMED: _RANDOM = {`RANDOM,`RANDOM,`RANDOM,`RANDOM};

    ; NONE:  reg [127:0] _RANDOM;
    ; NONE:  _RANDOM = {`RANDOM,`RANDOM,`RANDOM,`RANDOM};

    ; ALL:       _r = _RANDOM[32:0];
    ; NAMED-NOT: _r = {{.*}};
    ; NONE-NOT:  _r = {{.*}};
    reg _r: UInt<33>, clock
    _r <= d

    ; ALL:       r = _RANDOM[65:33];
    ; NAMED:     r = _RANDOM[65:33];
    ; NONE-NOT:  r = {{.*}};
    reg r: UInt<33>, clock
    r <= d

    ; ALL:       s = _RANDOM[98:66];
    ; NAMED:     s = _RANDOM[98:66];
    ; NONE:      s = _RANDOM[98:66];
    reg s: UInt<33>, clock
    s <= d

    q <= s

    inst i0 of TwoLargeRegisters
    i0.clock <= clock
    i0.d <= d0
    q0 <= i0.q

    inst i1 of OneReallyLargeRegister
    i1.clock <= clock
    i1.d <= d1
    q1 <= i1.q

  module TwoLargeRegisters:
    input clock: Clock
    input d: UInt<60000>
    output q: UInt<60000>

    ; ALL: reg [59999:0] _RANDOM;
    ; ALL: reg [59999:0] _RANDOM_0;

    reg r0: UInt<60000>, clock
    reg r1: UInt<60000>, clock

    r0 <= d
    r1 <= r0
    q <= r1

  module OneReallyLargeRegister:
    input clock: Clock
    input d: UInt<120000>
    output q: UInt<120000>

    ; ALL: reg [119999:0] _RANDOM;

    reg r: UInt<120000>, clock

    r <= d
    q <= r
