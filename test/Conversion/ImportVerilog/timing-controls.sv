// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// CHECK-LABEL: moore.module @RepeatedEventControl
module RepeatedEventControl;
  int a, b;
  bit clk;
  bit [31:0] n;
  real r;

  // Intra-assignment repeated event control with a signed count. The
  // right-hand side is evaluated first, then the event control is awaited
  // count times, then the assignment takes effect. A count of zero or less
  // skips the event control entirely.
  initial begin
    // CHECK: [[RHS:%.+]] = moore.read %b
    // CHECK: [[COUNT:%.+]] = moore.constant 3 : i32
    // CHECK: cf.br ^[[CHECK:bb[0-9]+]]([[COUNT]] : !moore.i32)
    // CHECK: ^[[CHECK]]([[REMAINING:%.+]]: !moore.i32):
    // CHECK: [[ZERO:%.+]] = moore.constant 0 : i32
    // CHECK: [[CMP:%.+]] = moore.sgt [[REMAINING]], [[ZERO]] : i32 -> i1
    // CHECK: [[COND:%.+]] = moore.to_builtin_int [[CMP]] : i1
    // CHECK: cf.cond_br [[COND]], ^[[BODY:bb[0-9]+]], ^[[EXIT:bb[0-9]+]]
    // CHECK: ^[[BODY]]:
    // CHECK: moore.wait_event {
    // CHECK:   [[CLK:%.+]] = moore.read %clk
    // CHECK:   moore.detect_event posedge [[CLK]]
    // CHECK: }
    // CHECK: [[ONE:%.+]] = moore.constant 1 : i32
    // CHECK: [[NEXT:%.+]] = moore.sub [[REMAINING]], [[ONE]] : i32
    // CHECK: cf.br ^[[CHECK]]([[NEXT]] : !moore.i32)
    // CHECK: ^[[EXIT]]:
    // CHECK: moore.blocking_assign %a, [[RHS]]
    a = repeat (3) @(posedge clk) b;
  end

  // An unsigned count uses an unsigned comparison.
  initial begin
    // CHECK: moore.ugt
    // CHECK: moore.wait_event
    // CHECK: moore.detect_event negedge
    // CHECK: moore.blocking_assign %a
    a = repeat (n) @(negedge clk) b;
  end

  // A floating count converts to a signed integer and uses a signed
  // comparison, so a negative count skips the event control.
  initial begin
    // CHECK: moore.sgt
    // CHECK: moore.wait_event
    // CHECK: moore.detect_event posedge
    // CHECK: moore.blocking_assign %a
    a = repeat (r) @(posedge clk) b;
  end
endmodule
