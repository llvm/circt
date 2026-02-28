; RUN: firtool %s --format=fir --ir-sv | FileCheck %s
FIRRTL version 4.0.0

circuit PlusArgTest:
  ; CHECK-LABEL: @PlusArgTest
  public module PlusArgTest :
    output foo_found : UInt<1>
    output bar_found : UInt<1>
    output bar_result : UInt<32>

    connect foo_found, intrinsic(circt_plusargs_test<FORMAT="foo"> : UInt<1>)

    node bar = intrinsic(circt_plusargs_value<FORMAT="foo=%d"> : { found : UInt<1>, result : UInt<32> })
    connect bar_found, bar.found
    connect bar_result, bar.result

    ; CHECK:      [[FORMAT_FOO:%.+]] = sv.constantStr "foo"
    ; CHECK-NEXT: [[FOUND_FOO_REG:%.+]] = sv.reg : !hw.inout<i1>
    ; CHECK-NEXT: sv.initial {
    ; CHECK-NEXT:   [[FOUND_FOO_VAL:%.+]] = sv.system "test$plusargs"([[FORMAT_FOO]]) : (!hw.string) -> i1
    ; CHECK-NEXT:   sv.bpassign [[FOUND_FOO_REG]], [[FOUND_FOO_VAL]] : i1
    ; CHECK-NEXT: }
    ; CHECK-NEXT: [[FOUND_FOO:%.+]] = sv.read_inout [[FOUND_FOO_REG]] : !hw.inout<i1>

    ; CHECK:      [[BAR_VALUE:%.+]] = sv.wire : !hw.inout<i32>
    ; CHECK-NEXT: [[BAR_FOUND:%.+]] = sv.wire : !hw.inout<i1>
    ; CHECK-NEXT: sv.ifdef @SYNTHESIS {
    ; CHECK-NEXT:   %z_i32 = sv.constantZ : i32
    ; CHECK-NEXT:   sv.assign [[BAR_VALUE]], %z_i32 {sv.attributes = [#sv.attribute<"This dummy assignment exists to avoid undriven lint warnings (e.g., Verilator UNDRIVEN).", emitAsComment>]} : i32
    ; CHECK-NEXT:   sv.assign [[BAR_FOUND]], %false : i1
    ; CHECK-NEXT: } else {
    ; CHECK-NEXT:   [[FOUND_REG:%.*]] = sv.reg : !hw.inout<i32>
    ; CHECK-NEXT:   [[VALUE_REG:%.*]] = sv.reg : !hw.inout<i32>
    ; CHECK-NEXT:   sv.initial {
    ; CHECK-NEXT:     [[FORMAT_BAR:%.+]] = sv.constantStr "foo=%d"
    ; CHECK-NEXT:     [[PLUSARG_FOUND:%.+]] = sv.system "value$plusargs"([[FORMAT_BAR]], [[VALUE_REG]]) : (!hw.string, !hw.inout<i32>) -> i32
    ; CHECK-NEXT:     sv.bpassign [[FOUND_REG]], [[PLUSARG_FOUND]]
    ; CHECK-NEXT:   }
    ; CHECK-NEXT:   [[FOUND_READ:%.*]] = sv.read_inout [[FOUND_REG]] : !hw.inout<i32>
    ; CHECK-NEXT:   [[VALUE_READ:%.*]] = sv.read_inout [[VALUE_REG]] : !hw.inout<i32>
    ; CHECK-NEXT:   [[FOUND:%.*]] = comb.icmp ceq [[FOUND_READ]], %c1_i32 : i32
    ; CHECK-NEXT:   sv.assign [[BAR_FOUND]], [[FOUND]] : i1
    ; CHECK-NEXT:   sv.assign [[BAR_VALUE]], [[VALUE_READ]] : i32
    ; CHECK-NEXT: }
    ; CHECK-NEXT: [[BAR_FOUND_READ:%.*]] = sv.read_inout [[BAR_FOUND]]
    ; CHECK-NEXT: [[BAR_VALUE_READ:%.*]] = sv.read_inout [[BAR_VALUE]]
    ; CHECK: [[FOUND_FOO]], [[BAR_FOUND_READ]], [[BAR_VALUE_READ]] : i1, i1, i32
