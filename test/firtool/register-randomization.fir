; RUN: firtool -preserve-values=all -verilog %s | FileCheck %s --check-prefix=ALL
; RUN: firtool -preserve-values=named -verilog %s | FileCheck %s --check-prefix=NAMED
; RUN: firtool -preserve-values=none -verilog %s | FileCheck %s --check-prefix=NONE

circuit Foo:
  module Foo:
    input clock: Clock
    input d: UInt<33>
    output q: UInt<33>

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
