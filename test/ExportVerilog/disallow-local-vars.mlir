// RUN: circt-translate --export-verilog %s | FileCheck %s
// RUN: circt-translate --lowering-options=disallowLocalVariables --export-verilog %s | FileCheck %s --check-prefix=DISALLOW

// This checks ExportVerilog's support for "disallowLocalVariables" which
// prevents emitting 'automatic logic' and other local declarations.

// CHECK-LABEL: module side_effect_expr
// DISALLOW-LABEL: module side_effect_expr
hw.module @side_effect_expr(%clock: i1) -> (%a: i1, %a2: i1) {

  // DISALLOW: reg [[SE_REG:[_A-Za-z0-9]+]];

  // DISALLOW: wire [[COND:[_A-Za-z0-9]+]] = INLINE_OK;

  // CHECK:    always @(posedge clock)
  // DISALLOW: always @(posedge clock)
  sv.always posedge %clock  {
    %0 = sv.verbatim.expr "INLINE_OK" : () -> i1

    // This shouldn't be pushed into a reg.
    // CHECK: if (INLINE_OK)
    // DISALLOW: if ([[COND]])
    sv.if %0  {
      sv.fatal
    }

    // This should go through a reg when in "disallow" mode.
    // CHECK: if (SIDE_EFFECT)
    // DISALLOW: [[SE_REG]] = SIDE_EFFECT;
    // DISALLOW: if ([[SE_REG]])
    %1 = sv.verbatim.expr.se "SIDE_EFFECT" : () -> i1
    sv.if %1  {
      sv.fatal
    }
  }

  // Top level things should go unmodified.
  %2 = sv.verbatim.expr "NO_SE" : () -> i1
  %3 = sv.verbatim.expr.se "YES_SE" : () -> i1

  // CHECK: assign a = NO_SE;
  // CHECK: assign a2 = YES_SE;
  // DISALLOW: assign a = NO_SE;
  // DISALLOW: assign a2 = YES_SE;
  hw.output %2, %3: i1, i1
}

// CHECK-LABEL: module hoist_expressions
// DISALLOW-LABEL: module hoist_expressions
hw.module @hoist_expressions(%clock: i1, %x: i8, %y: i8, %z: i8) {
  // DISALLOW: wire [7:0] [[ADD:[_A-Za-z0-9]+]] = x + y;
  // DISALLOW: wire [[EQ:[_A-Za-z0-9]+]] = [[ADD]] == z;
  // DISALLOW: wire [7:0] [[MUL:[_A-Za-z0-9]+]] = [[ADD]] * z;

  // CHECK:    always @(posedge clock)
  // DISALLOW: always @(posedge clock)
  sv.always posedge %clock  {
    %0 = comb.add %x, %y: i8
    %1 = comb.icmp eq %0, %z : i8

    // This shouldn't be touched.
    // CHECK: if (_T == z) begin
    // DISALLOW: if ([[EQ]]) begin
    sv.if %1  {
      // CHECK: $fwrite(32'h80000002, "Hi %x\n", _T * z);
      // DISALLOW: $fwrite(32'h80000002, "Hi %x\n", [[MUL]]);
      %2 = comb.mul %0, %z : i8
      sv.fwrite "Hi %x\0A"(%2) : i8
      sv.fatal
    }
  }

  // Check out wires.
  // CHECK: assign myWire = x;
  // DISALLOW: assign myWire = x;
  %myWire = sv.wire : !hw.inout<i8>
  sv.assign %myWire, %x : i8

 // DISALLOW: wire [[COND:[_A-Za-z0-9]+]] = x + myWire == z;

  // CHECK: always @(posedge clock)
  // DISALLOW: always @(posedge clock)
  sv.always posedge %clock  {
    %wireout = sv.read_inout %myWire : !hw.inout<i8>
    %3 = comb.add %x, %wireout: i8
    %4 = comb.icmp eq %3, %z : i8
    // CHECK: if (x + myWire == z)
    // DISALLOW: if ([[COND]])
    sv.if %4  {
      sv.fatal
    }
 }

  hw.output
}
