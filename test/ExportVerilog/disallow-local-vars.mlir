// RUN: circt-translate --export-verilog %s | FileCheck %s
// RUN: circt-translate --lowering-options=disallowLocalVariables --export-verilog %s | FileCheck %s --check-prefix=DISALLOW

// This checks ExportVerilog's support for "disallowLocalVariables" which
// prevents emitting 'automatic logic' and other local declarations.

// CHECK-LABEL: module side_effect_expr
// DISALLOW-LABEL: module side_effect_expr
hw.module @side_effect_expr(%clock: i1) -> (%a: i1, %a2: i1) {

  // DISALLOW: reg [[SE_REG:[_A-Za-z0-9]+]];

  // CHECK:    always @(posedge clock)
  // DISALLOW: always @(posedge clock)
  sv.always posedge %clock  {
    %0 = sv.verbatim.expr "INLINE_OK" : () -> i1

    // This shouldn't be touched.
    // CHECK: if (INLINE_OK)
    // DISALLOW: if (INLINE_OK)
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
