; RUN: firtool %s --format=fir | FileCheck %s

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


  module PlusArgTest :
    output foo_found : UInt<1>
    output bar_found : UInt<1>
    output bar_result : UInt<32>


    inst foo of PlusArgFooTest
    foo_found <= foo.found

    inst bar of PlusArgBarValue
    bar_found <= bar.found
    bar_result <= bar.result

  ; CHECK:       reg        [[FOO_FOUND:.+]];
  ; CHECK-NEXT:  initial
  ; CHECK-NEXT:    [[FOO_FOUND]] = $test$plusargs("foo");

  ; CHECK-NEXT:  reg [31:0] [[BAR_RESULT:.+]];
  ; CHECK-NEXT:  reg        [[BAR_FOUND:.+]];
  ; CHECK-NEXT:  `ifdef SYNTHESIS
  ; CHECK-NEXT:    /* This dummy assignment exists to avoid undriven lint warnings (e.g., Verilator UNDRIVEN). */
  ; CHECK-NEXT:    assign [[BAR_RESULT]] = 32'bz;
  ; CHECK-NEXT:    assign [[BAR_FOUND]] = 1'h0;
  ; CHECK-NEXT:  `else  // SYNTHESIS
  ; CHECK-NEXT:    initial
  ; CHECK-NEXT:      [[BAR_FOUND]] = |$value$plusargs("foo=%d", [[BAR_RESULT]]);
  ; CHECK-NEXT:  `endif // SYNTHESIS

  ; CHECK-NEXT:  assign foo_found = [[FOO_FOUND]];
  ; CHECK-NEXT:  assign bar_found = [[BAR_FOUND]];
  ; CHECK-NEXT:  assign bar_result = [[BAR_RESULT]];
