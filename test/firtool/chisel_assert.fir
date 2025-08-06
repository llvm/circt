; RUN: firtool %s | FileCheck %s

FIRRTL version 4.0.0

circuit ChiselVerif:
  ; CHECK: module ChiselVerif
  public module ChiselVerif:
    input clock: Clock
    input cond: UInt<1>
    input enable: UInt<1>
    ; CHECK:  wire [[GEN:.+]] = ~enable | cond;
    ; CHECK: assert property
    ; CHECK-NOT: $error
    intrinsic(circt_chisel_assert, clock, cond, enable)

    ; CHECK: `ifdef MACRO_GUARD
    ; CHECK-NEXT: `ifdef ASDF
    ; CHECK: label_for_assert_with_format_string
    ; CHECK: assert property
    ; CHECK: "message: %d"
    ; CHECK: $sampled(cond)
    intrinsic(circt_chisel_assert<format = "message: %d",
                                  label = "label for assert with format string",
                                  guards = "MACRO_GUARD;ASDF">,
              clock, cond, enable, cond)

    ; Special if-else-fatal pattern, assert-like.
    ; No guards or labels for normal emission flow.
    ; CHECK: $error("ief: %d"
    ; CHECK: $fatal
    intrinsic(circt_chisel_ifelsefatal<format = "ief: %d",
                                       label = "label for ifelsefatal assert",
                                       guards = "MACRO_GUARD;ASDF">,
              clock, cond, enable, enable)

    ; CHECK: label_for_assume
    ; CHECK: assume property
    ; CHECK: "text: %d"
    ; CHECK: $sampled(enable)
    intrinsic(circt_chisel_assume<format = "text: %d",
                                  label = "label for assume">,
              clock, cond, enable, enable)

    ; CHECK: label_for_cover
    ; CHECK: cover property
    intrinsic(circt_chisel_cover<label = "label for cover">,
              clock, cond, enable)
