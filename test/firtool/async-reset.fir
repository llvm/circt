; RUN: firtool --split-input-file %s | FileCheck %s

FIRRTL version 3.3.0
; CHECK-LABEL: module test(
circuit test :%[[{
  "class":"circt.FullResetAnnotation",
    "target":"~test|test>reset",
    "resetType":"async"
  }]]
  module test :
    input clock : Clock
    input reset : AsyncReset
    input in : { foo : UInt<8>, bar : UInt<8>}
    output out : { foo : UInt<8>, bar : UInt<8>}

    wire reg1_w : { foo : UInt<8>, bar : UInt<8>}
    invalidate reg1_w.bar
    invalidate reg1_w.foo
    ; CHECK: reg1_foo <= 8'hC;
    ; CHECK: reg1_bar <= 8'h0;
    connect reg1_w.foo, UInt<8>(0hc)
    invalidate reg1_w.bar
    ; CHECK: reg1_foo = 8'hC;
    ; CHECK: reg1_bar = 8'h0;
    regreset reg1 : { foo : UInt<8>, bar : UInt<8>}, clock, reset, reg1_w
    wire reg2 : { foo : UInt<8>, bar : UInt<8>}
    connect reg1, in
    connect reg2, reg1
    connect out, reg2

;// -----
; CHECK-LABEL: module test_wire(
FIRRTL version 3.3.0
circuit test_wire :%[[{
  "class":"circt.FullResetAnnotation",
    "target":"~test_wire|test_wire>reset",
    "resetType":"async"
  }]]
  module test_wire :
    input clock : Clock
    input r : AsyncReset
    input in : { foo : UInt<8>, bar : UInt<8>}
    output out : { foo : UInt<8>, bar : UInt<8>}

    node reset = r

    wire reg1_w : { foo : UInt<8>, bar : UInt<8>}
    invalidate reg1_w.bar
    invalidate reg1_w.foo
    ; CHECK: reg1_foo <= 8'hC;
    ; CHECK: reg1_bar <= 8'h0;
    connect reg1_w.foo, UInt<8>(0hc)
    invalidate reg1_w.bar
    ; CHECK: reg1_foo = 8'hC;
    ; CHECK: reg1_bar = 8'h0;
    regreset reg1 : { foo : UInt<8>, bar : UInt<8>}, clock, reset, reg1_w
    wire reg2 : { foo : UInt<8>, bar : UInt<8>}
    connect reg1, in
    connect reg2, reg1
    connect out, reg2

;// -----
; CHECK-LABEL: module test_sync(
FIRRTL version 3.3.0
circuit test_sync :%[[{
  "class":"circt.FullResetAnnotation",
    "target":"~test_sync|test_sync>reset",
    "resetType":"sync"
  }]]
  module test_sync :
    input clock : Clock
    input reset : UInt<1>
    input in : { foo : UInt<8>, bar : UInt<8>}
    output out : { foo : UInt<8>, bar : UInt<8>}

    wire reg1_w : { foo : UInt<8>, bar : UInt<8>}
    invalidate reg1_w.bar
    invalidate reg1_w.foo
    connect reg1_w.foo, UInt<8>(0hc)
    invalidate reg1_w.bar
    ; CHECK:      always @(posedge clock) begin
    ; CHECK-NEXT:   if (reset) begin
    ; CHECK-NEXT:     reg1_foo <= 8'hC;
    ; CHECK-NEXT:     reg1_bar <= 8'h0;
    regreset reg1 : { foo : UInt<8>, bar : UInt<8>}, clock, reset, reg1_w
    wire reg2 : { foo : UInt<8>, bar : UInt<8>}
    connect reg1, in
    connect reg2, reg1
    connect out, reg2

;// -----
; CHECK-LABEL: module test_wire_sync(
FIRRTL version 3.3.0
circuit test_wire_sync :%[[{
  "class":"circt.FullResetAnnotation",
    "target":"~test_wire_sync|test_wire_sync>reset",
    "resetType":"sync"
  }]]
  module test_wire_sync :
    input clock : Clock
    input rst : UInt<1>
    input in : { foo : UInt<8>, bar : UInt<8>}
    output out : { foo : UInt<8>, bar : UInt<8>}

    node reset = rst

    wire reg1_w : { foo : UInt<8>, bar : UInt<8>}
    invalidate reg1_w.bar
    invalidate reg1_w.foo
    connect reg1_w.foo, UInt<8>(0hc)
    invalidate reg1_w.bar
    ; CHECK:      always @(posedge clock) begin
    ; CHECK-NEXT:   if (rst) begin
    ; CHECK-NEXT:     reg1_foo <= 8'hC;
    ; CHECK-NEXT:     reg1_bar <= 8'h0;
    regreset reg1 : { foo : UInt<8>, bar : UInt<8>}, clock, reset, reg1_w
    wire reg2 : { foo : UInt<8>, bar : UInt<8>}
    connect reg1, in
    connect reg2, reg1
    connect out, reg2
