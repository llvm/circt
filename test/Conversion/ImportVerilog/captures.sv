// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// Recursive function with a capture
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @RecursiveCaptureFunction() {
module RecursiveCaptureFunction;
  // CHECK: %captureMe = moore.variable : <i32>
  int captureMe;
  int r;
  initial begin
    // CHECK: func.call @fact({{.*}}, %captureMe) : (!moore.i32, !moore.ref<i32>) -> !moore.i32
    r = fact(5);
  end

  // CHECK: func.func private @fact(%arg0: !moore.i32, %arg1: !moore.ref<i32>) -> !moore.i32 {
  function int fact(input int n);
    // CHECK: moore.read %arg1 : ref<i32>
    if (n <= 1) return captureMe;
    // CHECK: call @fact({{.*}}, %arg1) : (!moore.i32, !moore.ref<i32>) -> !moore.i32
    return n * fact(n - 1);
  endfunction
endmodule

//===----------------------------------------------------------------------===//
// Task capturing a signal used in an event control expression
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @CaptureInEventControl
module CaptureInEventControl;
  // CHECK: %clk = moore.variable : <l1>
  logic clk;
  // CHECK: %data = moore.variable : <i32>
  int data;

  initial begin
    // CHECK: moore.call_coroutine @waitForClk(%clk)
    waitForClk();
    // CHECK: moore.call_coroutine @readOnClk(%clk, %data)
    readOnClk();
  end

  // CHECK: moore.coroutine private @waitForClk(%arg0: !moore.ref<l1>)
  task automatic waitForClk;
    // CHECK: moore.read %arg0 : ref<l1>
    // CHECK: moore.detect_event posedge
    @(posedge clk);
  endtask

  // CHECK: moore.coroutine private @readOnClk(%arg0: !moore.ref<l1>, %arg1: !moore.ref<i32>)
  task automatic readOnClk;
    int result;
    // CHECK: moore.detect_event posedge
    @(posedge clk);
    result = data;
  endtask
endmodule

//===----------------------------------------------------------------------===//
// Transitive capture: wrapper -> inner, inner captures x
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @TransitiveCapture
module TransitiveCapture;
  // CHECK: %x = moore.variable : <i32>
  int x;

  initial begin
    // CHECK: moore.call_coroutine @wrapper({{.*}}, %x)
    wrapper(42);
  end

  // CHECK: moore.coroutine private @inner(%arg0: !moore.i32, %arg1: !moore.ref<i32>)
  task automatic inner(int val);
    x = val;
  endtask

  // CHECK: moore.coroutine private @wrapper(%arg0: !moore.i32, %arg1: !moore.ref<i32>)
  task automatic wrapper(int val);
    // CHECK: moore.call_coroutine @inner(%arg0, %arg1)
    inner(val);
  endtask
endmodule

//===----------------------------------------------------------------------===//
// Compile-time constants should not be captured
//===----------------------------------------------------------------------===//

// Parameter referenced in a function that can't be constant-folded.
// CHECK-LABEL: func.func private @scale() -> !moore.i32
module ParamNotCaptured;
  parameter MAX = 256;
  function automatic int scale();
    return $random % MAX;
  endfunction
  initial $display("%0d", scale());
endmodule

// Enum value referenced in a task.
// CHECK-LABEL: moore.coroutine private @check(%arg0: !moore.l2)
module EnumNotCaptured;
  typedef enum logic [1:0] { A, B, C } state_t;
  task automatic check(state_t s);
    if (s === B)
      $display("PASSED");
  endtask
  initial check(B);
endmodule

// Localparam referenced in a function via non-constant expression.
// CHECK-LABEL: func.func private @get_rand() -> !moore.i32
module LocalparamNotCaptured;
  localparam OFFSET = 42;
  function automatic int get_rand();
    return $random + OFFSET;
  endfunction
  initial $display("%0d", get_rand());
endmodule

// Class parameter referenced in a method.
// CHECK: func.func private @"ClassParamNotCaptured::C::get"(%arg0: !moore.class<@"ClassParamNotCaptured::C">) -> !moore.i32
module ClassParamNotCaptured;
  class C #(int N = 10);
    function automatic int get();
      return $random % N;
    endfunction
  endclass

  initial begin
    C #(5) c = new;
    $display("%0d", c.get());
  end
endmodule

//===----------------------------------------------------------------------===//
// Function and task capturing a net
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @CaptureNet
module CaptureNet(input logic a, b);
  // CHECK: %w = moore.net wire {{.*}} : <l2>
  wire [1:0] w = {a, b};
  // CHECK: %r = moore.variable : <l1>
  logic r;

  // CHECK: func.call @netFunc(%w) : (!moore.net<l2>) -> !moore.l1
  always_comb r = netFunc();
  // CHECK: moore.call_coroutine @netTask(%r, %w) : (!moore.ref<l1>, !moore.net<l2>) -> ()
  initial netTask();

  // CHECK: func.func private @netFunc(%arg0: !moore.net<l2>) -> !moore.l1 {
  function automatic logic netFunc();
    // CHECK: moore.read %arg0 : net<l2>
    return w[0] ^ w[1];
  endfunction

  // CHECK: moore.coroutine private @netTask(%arg0: !moore.ref<l1>, %arg1: !moore.net<l2>) {
  task automatic netTask();
    // CHECK: moore.read %arg1 : net<l2>
    r = w[0];
  endtask
endmodule
