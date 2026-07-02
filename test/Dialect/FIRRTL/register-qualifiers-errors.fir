; RUN: circt-translate -import-firrtl -verify-diagnostics -split-input-file %s

; Fail-closed: every register must state its clock edge, and every reset-bearing
; register its reset kind and polarity, via the required inline qualifiers. A
; missing qualifier is a parse error, never a silently-stamped default.

FIRRTL version 4.0.0
circuit MissingClockEdge :
  public module MissingClockEdge :
    input clock : Clock
    input d : UInt<8>
    output q : UInt<8>
    ; expected-error @below {{expected 'posedge' or 'negedge' clock-edge qualifier before the register clock}}
    reg r : UInt<8>, clock
    connect r, d
    connect q, r

;// -----

FIRRTL version 4.0.0
circuit MissingResetKind :
  public module MissingResetKind :
    input clock : Clock
    input reset : Reset
    input d : UInt<8>
    output q : UInt<8>
    ; expected-error @below {{expected 'sync' or 'async' reset-kind qualifier before the register reset}}
    regreset r : UInt<8>, posedge clock, activehigh reset, UInt<8>(0)
    connect r, d
    connect q, r

;// -----

FIRRTL version 4.0.0
circuit MissingResetPolarity :
  public module MissingResetPolarity :
    input clock : Clock
    input reset : Reset
    input d : UInt<8>
    output q : UInt<8>
    ; expected-error @below {{expected 'activehigh' or 'activelow' reset-polarity qualifier before the register reset}}
    regreset r : UInt<8>, posedge clock, sync reset, UInt<8>(0)
    connect r, d
    connect q, r

;// -----

FIRRTL version 4.0.0
circuit AsyncResetType :
  public module AsyncResetType :
    ; expected-error @below {{the 'AsyncReset' type is not supported}}
    input r : AsyncReset
    output q : UInt<8>
    connect q, UInt<8>(0)
