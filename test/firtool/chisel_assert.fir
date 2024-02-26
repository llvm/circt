; RUN: firtool %s | FileCheck %s

FIRRTL version 4.0.0

circuit ChiselVerif:
  intmodule AssertAssume:
    input clock: Clock
    input predicate: UInt<1>
    input enable: UInt<1>
    intrinsic = circt_chisel_assert_assume

  intmodule AssertAssumeFormat:
    input clock: Clock
    input predicate: UInt<1>
    input enable: UInt<1>
    input val: UInt<1>
    intrinsic = circt_chisel_assert_assume
    parameter format = "message: %d"
    parameter label = "label for assert with format string"
    parameter guards = "MACRO_GUARD;ASDF"

  intmodule IfElseFatalFormat:
    input clock: Clock
    input predicate: UInt<1>
    input enable: UInt<1>
    input val: UInt<1>
    intrinsic = circt_chisel_ifelsefatal
    parameter format = "ief: %d"
    ; In normal emission these are unused, but allow them to be set for now.
    parameter label = "label for ifelsefatal assert"
    parameter guards = "MACRO_GUARD;ASDF"

  intmodule Assume:
    input clock: Clock
    input predicate: UInt<1>
    input enable: UInt<1>
    input val: UInt<1>
    intrinsic = circt_chisel_assume
    parameter format = "text: %d"
    parameter label = "label for assume"

  intmodule CoverLabel:
    input clock: Clock
    input predicate: UInt<1>
    input enable: UInt<1>
    intrinsic = circt_chisel_cover
    parameter label = "label for cover"

  ; CHECK: module ChiselVerif
  public module ChiselVerif:
    input clock: Clock
    input cond: UInt<1>
    input enable: UInt<1>

    ; CHECK: assert property
    ; CHECK-NOT: $error
    ; CHECK: PROPERTY_AS_CONSTRAINT
    ; CHECK: assume
    inst assert of AssertAssume
    connect assert.clock, clock
    connect assert.predicate, cond
    connect assert.enable, enable

    ; CHECK: `ifdef MACRO_GUARD
    ; CHECK-NEXT: `ifdef ASDF
    ; CHECK: label_for_assert_with_format_string
    ; CHECK: assert property
    ; CHECK: "message: %d"
    ; CHECK: $sampled(cond)
    ; CHECK: PROPERTY_AS_CONSTRAINT
    ; CHECK: assume
    inst assertFormat of AssertAssumeFormat
    connect assertFormat.clock, clock
    connect assertFormat.predicate, cond
    connect assertFormat.enable, enable
    connect assertFormat.val, cond

    ; Special if-else-fatal pattern, assert-like.
    ; No guards or labels for normal emission flow.
    ; CHECK: $error("ief: %d"
    ; CHECK: $fatal
    inst ief of IfElseFatalFormat
    connect ief.clock, clock
    connect ief.predicate, cond
    connect ief.enable, enable
    connect ief.val, enable

    ; CHECK: label_for_assume
    ; CHECK: assume property
    ; CHECK: "text: %d"
    ; CHECK: $sampled(enable)
    inst assume of Assume
    connect assume.clock, clock
    connect assume.predicate, cond
    connect assume.enable, enable
    connect assume.val, enable

    ; CHECK: label_for_cover
    ; CHECK: cover property
    inst cover of CoverLabel
    connect cover.clock, clock
    connect cover.predicate, cond
    connect cover.enable, enable
