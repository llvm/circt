; RUN: firtool %s --format=fir --ir-sv | FileCheck %s

circuit PlusArgTest:
  intmodule PlusArgFooTest :
    output found : UInt<1>
    intrinsic = circt_plusargs_test
    parameter FORMAT = "foo"

  intmodule PlusArgBarValue :
    output found : UInt<1>
    output result : UInt<32>
    intrinsic = circt_plusargs_value
    parameter FORMAT = "foo=%d"


  ; CHECK-LABEL: @PlusArgTest
  module PlusArgTest :
    output foo_found : UInt<1>
    output bar_found : UInt<1>
    output bar_result : UInt<32>


    inst foo of PlusArgFooTest
    foo_found <= foo.found

    inst bar of PlusArgBarValue
    bar_found <= bar.found
    bar_result <= bar.result


    ; CHECK:      [[FORMAT_FOO:%.+]] = sv.constantStr "foo"
    ; CHECK-NEXT: [[FOUND_FOO_REG:%.+]] = sv.reg : !hw.inout<i1>
    ; CHECK-NEXT: sv.initial {
    ; CHECK-NEXT:   [[FOUND_FOO_VAL:%.+]] = sv.system "test$plusargs"([[FORMAT_FOO]]) : (!hw.string) -> i1
    ; CHECK-NEXT:   sv.bpassign [[FOUND_FOO_REG]], [[FOUND_FOO_VAL]] : i1
    ; CHECK-NEXT: }
    ; CHECK-NEXT: [[FOUND_FOO:%.+]] = sv.read_inout [[FOUND_FOO_REG]] : !hw.inout<i1>

    ; CHECK:      [[RESULT_BAR_REG:%.+]] = sv.reg : !hw.inout<i32>
    ; CHECK-NEXT: [[FOUND_BAR_REG:%.+]] = sv.reg : !hw.inout<i1>
    ; CHECK-NEXT: sv.ifdef @SYNTHESIS {
    ; CHECK-NEXT:   %z_i32 = sv.constantZ : i32
    ; CHECK-NEXT:   sv.assign [[RESULT_BAR_REG]], %z_i32 {sv.attributes = [#sv.attribute<"This dummy assignment exists to avoid undriven lint warnings (e.g., Verilator UNDRIVEN).", emitAsComment>]} : i32
    ; CHECK-NEXT:   sv.assign [[FOUND_BAR_REG]], %false : i1
    ; CHECK-NEXT: } else {
    ; CHECK-NEXT:   sv.initial {
    ; CHECK-NEXT:     [[FORMAT_BAR:%.+]] = sv.constantStr "foo=%d"
    ; CHECK-NEXT:     [[TMP_BAR:%.+]] = sv.system "value$plusargs"([[FORMAT_BAR]], [[RESULT_BAR_REG]]) : (!hw.string, !hw.inout<i32>) -> i32
    ; CHECK-NEXT:     [[FOUND_BAR_VAL:%.+]] = comb.icmp bin ne [[TMP_BAR]], %c0_i32 : i32
    ; CHECK-NEXT:     sv.bpassign [[FOUND_BAR_REG]], [[FOUND_BAR_VAL]] : i1
    ; CHECK-NEXT:   }
    ; CHECK-NEXT: }
    ; CHECK-NEXT: [[FOUND_BAR:%.+]] = sv.read_inout [[FOUND_BAR_REG]] : !hw.inout<i1>
    ; CHECK-NEXT: [[RESULT_BAR:%.+]] = sv.read_inout [[RESULT_BAR_REG]] : !hw.inout<i32>

    ; CHECK: [[FOUND_FOO]], [[FOUND_BAR]], [[RESULT_BAR]] : i1, i1, i32
