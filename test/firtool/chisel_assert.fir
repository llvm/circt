; RUN: firtool %s | FileCheck %s

FIRRTL version 4.0.0

circuit ChiselVerif:
  intmodule Assert:
    input clock: Clock
    input predicate: UInt<1>
    input enable: UInt<1>
    intrinsic = circt_chisel_assert
    parameter format = "testing"

  intmodule AssertFormat:
    input clock: Clock
    input predicate: UInt<1>
    input enable: UInt<1>
    input val: UInt<1>
    intrinsic = circt_chisel_assert
    parameter format = "message: %d"
    parameter label = "label for assert with format string"
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
  module ChiselVerif:
    input clock: Clock
    input cond: UInt<1>
    input enable: UInt<1>

    ; CHECK: assert property
    ; CHECk-SAME: "testing"
    inst assert of Assert
    connect assert.clock, clock
    connect assert.predicate, cond
    connect assert.enable, enable

    ; CHECK: `ifdef MACRO_GUARD
    ; CHECK-NEXT: `ifdef ASDF
    ; CHECK: label_for_assert_with_format_string
    ; CHECK: assert property
    ; CHECK: "message: %d"
    ; CHECK: $sampled(cond)
    inst assertFormat of AssertFormat
    connect assertFormat.clock, clock
    connect assertFormat.predicate, cond
    connect assertFormat.enable, enable
    connect assertFormat.val, cond

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
