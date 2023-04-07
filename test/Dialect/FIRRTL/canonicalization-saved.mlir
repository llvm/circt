
// CHECK-LABEL: firrtl.module @issue446
// CHECK-NEXT: [[TMP:%.+]] = firrtl.constant 0 : !firrtl.uint<1>
// CHECK-NEXT: firrtl.strictconnect %tmp10, [[TMP]] : !firrtl.uint<1>
firrtl.module @issue446(in %inp_1: !firrtl.sint<0>, out %tmp10: !firrtl.uint<1>) {
  %0 = firrtl.xor %inp_1, %inp_1 : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<0>
  firrtl.strictconnect %tmp10, %0 : !firrtl.uint<1>, !firrtl.uint<0>
}

// CHECK-LABEL: firrtl.module @xorUnsized
// CHECK-NEXT: %c0_ui = firrtl.constant 0 : !firrtl.uint
firrtl.module @xorUnsized(in %inp_1: !firrtl.sint, out %tmp10: !firrtl.uint) {
  %0 = firrtl.xor %inp_1, %inp_1 : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint
  firrtl.strictconnect %tmp10, %0 : !firrtl.uint, !firrtl.uint
}

// Sign casts must not be folded into unsized constants.
// CHECK-LABEL: firrtl.module @issue1118
firrtl.module @issue1118(out %z0: !firrtl.uint, out %z1: !firrtl.sint) {
  // CHECK: %0 = firrtl.asUInt %c4232_si : (!firrtl.sint) -> !firrtl.uint
  // CHECK: %1 = firrtl.asSInt %c4232_ui : (!firrtl.uint) -> !firrtl.sint
  // CHECK: firrtl.strictconnect %z0, %0 : !firrtl.uint, !firrtl.uint
  // CHECK: firrtl.strictconnect %z1, %1 : !firrtl.sint, !firrtl.sint
  %c4232_si = firrtl.constant 4232 : !firrtl.sint
  %c4232_ui = firrtl.constant 4232 : !firrtl.uint
  %0 = firrtl.asUInt %c4232_si : (!firrtl.sint) -> !firrtl.uint
  %1 = firrtl.asSInt %c4232_ui : (!firrtl.uint) -> !firrtl.sint
  firrtl.strictconnect %z0, %0 : !firrtl.uint, !firrtl.uint
  firrtl.strictconnect %z1, %1 : !firrtl.sint, !firrtl.sint
}

// CHECK-LABEL: firrtl.module @issue1139
firrtl.module @issue1139(out %z: !firrtl.uint<4>) {
  // CHECK-NEXT: %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
  // CHECK-NEXT: firrtl.strictconnect %z, %c0_ui4 : !firrtl.uint<4>
  %c4_ui4 = firrtl.constant 4 : !firrtl.uint<4>
  %c674_ui = firrtl.constant 674 : !firrtl.uint
  %0 = firrtl.dshr %c4_ui4, %c674_ui : (!firrtl.uint<4>, !firrtl.uint) -> !firrtl.uint<4>
  firrtl.strictconnect %z, %0 : !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @issue1142
firrtl.module @issue1142(in %cond: !firrtl.uint<1>, out %z: !firrtl.uint) {
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %c42_ui = firrtl.constant 42 : !firrtl.uint
  %c43_ui = firrtl.constant 43 : !firrtl.uint

  // Don't fold away constant selects if widths are unknown.
  // CHECK: %0 = firrtl.mux(%c0_ui1, %c42_ui, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  // CHECK: %1 = firrtl.mux(%c1_ui1, %c42_ui, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  %0 = firrtl.mux(%c0_ui1, %c42_ui, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  %1 = firrtl.mux(%c1_ui1, %c42_ui, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint

  // Don't fold nested muxes with same condition if widths are unknown.
  // CHECK: %2 = firrtl.mux(%cond, %c42_ui, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  // CHECK: %3 = firrtl.mux(%cond, %2, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  // CHECK: %4 = firrtl.mux(%cond, %c42_ui, %2) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  %2 = firrtl.mux(%cond, %c42_ui, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  %3 = firrtl.mux(%cond, %2, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  %4 = firrtl.mux(%cond, %c42_ui, %2) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint

  firrtl.strictconnect %z, %0 : !firrtl.uint, !firrtl.uint
  firrtl.strictconnect %z, %1 : !firrtl.uint, !firrtl.uint
  firrtl.strictconnect %z, %3 : !firrtl.uint, !firrtl.uint
  firrtl.strictconnect %z, %4 : !firrtl.uint, !firrtl.uint
}


// CHECK-LABEL: firrtl.module @Div
firrtl.module @Div(in %a: !firrtl.uint<4>,
                   out %b: !firrtl.uint<4>,
                   in %c: !firrtl.sint<4>,
                   out %d: !firrtl.sint<5>,
                   in %e: !firrtl.uint,
                   out %f: !firrtl.uint,
                   in %g: !firrtl.sint,
                   out %h: !firrtl.sint,
                   out %i: !firrtl.uint<4>) {

  // CHECK-DAG: [[ONE_i4:%.+]] = firrtl.constant 1 : !firrtl.uint<4>
  // CHECK-DAG: [[ONE_s5:%.+]] = firrtl.constant 1 : !firrtl.sint<5>
  // CHECK-DAG: [[ONE_i2:%.+]] = firrtl.constant 1 : !firrtl.uint
  // CHECK-DAG: [[ONE_s2:%.+]] = firrtl.constant 1 : !firrtl.sint

  // Check that 'div(e, e) -> 1' works for unknown UInt widths.
  // CHECK: firrtl.strictconnect %f, [[ONE_i2]]
  %2 = firrtl.div %e, %e : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
  firrtl.connect %f, %2 : !firrtl.uint, !firrtl.uint

  // Check that 'div(g, g) -> 1' works for unknown SInt widths.
  // CHECK: firrtl.strictconnect %h, [[ONE_s2]]
  %3 = firrtl.div %g, %g : (!firrtl.sint, !firrtl.sint) -> !firrtl.sint
  firrtl.connect %h, %3 : !firrtl.sint, !firrtl.sint

}

// CHECK-LABEL: @LEQWithConstLHS
// CHECK-NEXT: %c42_ui = firrtl.constant
// CHECK-NEXT: %e = firrtl.geq %a, %c42_ui
firrtl.module @LEQWithConstLHS(in %a: !firrtl.uint, out %b: !firrtl.uint<1>) {
  %0 = firrtl.constant 42 : !firrtl.uint
  %1 = firrtl.leq %0, %a {name = "e"} : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.strictconnect %b, %1 : !firrtl.uint<1>
}

// CHECK-LABEL: @LTWithConstLHS
// CHECK-NEXT: %c42_ui = firrtl.constant
// CHECK-NEXT: %0 = firrtl.gt %a, %c42_ui
firrtl.module @LTWithConstLHS(in %a: !firrtl.uint, out %b: !firrtl.uint<1>) {
  %0 = firrtl.constant 42 : !firrtl.uint
  %1 = firrtl.lt %0, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.strictconnect %b, %1 : !firrtl.uint<1>
}

// CHECK-LABEL: @GEQWithConstLHS
// CHECK-NEXT: %c42_ui = firrtl.constant
// CHECK-NEXT: %0 = firrtl.leq %a, %c42_ui
firrtl.module @GEQWithConstLHS(in %a: !firrtl.uint, out %b: !firrtl.uint<1>) {
  %0 = firrtl.constant 42 : !firrtl.uint
  %1 = firrtl.geq %0, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.strictconnect %b, %1 : !firrtl.uint<1>
}

// CHECK-LABEL: @GTWithConstLHS
// CHECK-NEXT: %c42_ui = firrtl.constant
// CHECK-NEXT: %0 = firrtl.lt %a, %c42_ui
firrtl.module @GTWithConstLHS(in %a: !firrtl.uint, out %b: !firrtl.uint<1>) {
  %0 = firrtl.constant 42 : !firrtl.uint
  %1 = firrtl.gt %0, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.strictconnect %b, %1 : !firrtl.uint<1>
}

// CHECK-LABEL: @CompareWithSelf
firrtl.module @CompareWithSelf(
  in %a: !firrtl.uint,
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant

  %0 = firrtl.leq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.strictconnect %y0, %0 : !firrtl.uint<1>
  // CHECK-NEXT: firrtl.strictconnect %y0, %c1_ui1

  %1 = firrtl.lt %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.strictconnect %y1, %1 : !firrtl.uint<1>
  // CHECK-NEXT: firrtl.strictconnect %y1, %c0_ui1

  %2 = firrtl.geq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.strictconnect %y2, %2 : !firrtl.uint<1>
  // CHECK-NEXT: firrtl.strictconnect %y2, %c1_ui1

  %3 = firrtl.gt %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.strictconnect %y3, %3 : !firrtl.uint<1>
  // CHECK-NEXT: firrtl.strictconnect %y3, %c0_ui1

  %4 = firrtl.eq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.strictconnect %y4, %4 : !firrtl.uint<1>
  // CHECK-NEXT: firrtl.strictconnect %y4, %c1_ui1

  %5 = firrtl.neq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.strictconnect %y5, %5 : !firrtl.uint<1>
  // CHECK-NEXT: firrtl.strictconnect %y5, %c0_ui1
}


// CHECK-LABEL: firrtl.module @PadMuxOperands
firrtl.module @PadMuxOperands(
  in %cond: !firrtl.uint<1>,
  in %ui: !firrtl.uint,
  in %ui11: !firrtl.uint<11>,
  in %ui17: !firrtl.uint<17>,
  out %z: !firrtl.uint
) {
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

  // Smaller operand should pad to result width.
  // CHECK: %0 = firrtl.pad %ui11, 17 : (!firrtl.uint<11>) -> !firrtl.uint<17>
  // CHECK: %1 = firrtl.mux(%cond, %0, %ui17) : (!firrtl.uint<1>, !firrtl.uint<17>, !firrtl.uint<17>) -> !firrtl.uint<17>
  // CHECK: %2 = firrtl.pad %ui11, 17 : (!firrtl.uint<11>) -> !firrtl.uint<17>
  // CHECK: %3 = firrtl.mux(%cond, %ui17, %2) : (!firrtl.uint<1>, !firrtl.uint<17>, !firrtl.uint<17>) -> !firrtl.uint<17>
  %0 = firrtl.mux(%cond, %ui11, %ui17) : (!firrtl.uint<1>, !firrtl.uint<11>, !firrtl.uint<17>) -> !firrtl.uint<17>
  %1 = firrtl.mux(%cond, %ui17, %ui11) : (!firrtl.uint<1>, !firrtl.uint<17>, !firrtl.uint<11>) -> !firrtl.uint<17>

  // Unknown result width should prevent padding.
  // CHECK: %4 = firrtl.mux(%cond, %ui11, %ui) : (!firrtl.uint<1>, !firrtl.uint<11>, !firrtl.uint) -> !firrtl.uint
  // CHECK: %5 = firrtl.mux(%cond, %ui, %ui11) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint<11>) -> !firrtl.uint
  %2 = firrtl.mux(%cond, %ui11, %ui) : (!firrtl.uint<1>, !firrtl.uint<11>, !firrtl.uint) -> !firrtl.uint
  %3 = firrtl.mux(%cond, %ui, %ui11) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint<11>) -> !firrtl.uint

  // Padding to equal width operands should enable constant-select folds.
  // CHECK: %6 = firrtl.pad %ui11, 17 : (!firrtl.uint<11>) -> !firrtl.uint<17>
  // CHECK: %7 = firrtl.pad %ui11, 17 : (!firrtl.uint<11>) -> !firrtl.uint<17>
  // CHECK: firrtl.strictconnect %z, %ui17 : !firrtl.uint, !firrtl.uint<17>
  // CHECK: firrtl.strictconnect %z, %6 : !firrtl.uint, !firrtl.uint<17>
  // CHECK: firrtl.strictconnect %z, %7 : !firrtl.uint, !firrtl.uint<17>
  // CHECK: firrtl.strictconnect %z, %ui17 : !firrtl.uint, !firrtl.uint<17>
  %4 = firrtl.mux(%c0_ui1, %ui11, %ui17) : (!firrtl.uint<1>, !firrtl.uint<11>, !firrtl.uint<17>) -> !firrtl.uint<17>
  %5 = firrtl.mux(%c0_ui1, %ui17, %ui11) : (!firrtl.uint<1>, !firrtl.uint<17>, !firrtl.uint<11>) -> !firrtl.uint<17>
  %6 = firrtl.mux(%c1_ui1, %ui11, %ui17) : (!firrtl.uint<1>, !firrtl.uint<11>, !firrtl.uint<17>) -> !firrtl.uint<17>
  %7 = firrtl.mux(%c1_ui1, %ui17, %ui11) : (!firrtl.uint<1>, !firrtl.uint<17>, !firrtl.uint<11>) -> !firrtl.uint<17>

  firrtl.strictconnect %z, %0 : !firrtl.uint, !firrtl.uint<17>
  firrtl.strictconnect %z, %1 : !firrtl.uint, !firrtl.uint<17>
  firrtl.strictconnect %z, %2 : !firrtl.uint, !firrtl.uint
  firrtl.strictconnect %z, %3 : !firrtl.uint, !firrtl.uint
  firrtl.strictconnect %z, %4 : !firrtl.uint, !firrtl.uint<17>
  firrtl.strictconnect %z, %5 : !firrtl.uint, !firrtl.uint<17>
  firrtl.strictconnect %z, %6 : !firrtl.uint, !firrtl.uint<17>
  firrtl.strictconnect %z, %7 : !firrtl.uint, !firrtl.uint<17>
}

// CHECK-LABEL: firrtl.module @issue1116
firrtl.module @issue1116(out %z: !firrtl.uint<1>) {
  %c844336_ui = firrtl.constant 844336 : !firrtl.uint
  %c161_ui8 = firrtl.constant 161 : !firrtl.uint<8>
  %0 = firrtl.leq %c844336_ui, %c161_ui8 : (!firrtl.uint, !firrtl.uint<8>) -> !firrtl.uint<1>
  // CHECK: firrtl.strictconnect %z, %c0_ui1
  firrtl.strictconnect %z, %0 : !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @regsyncreset_no
firrtl.module @regsyncreset_no(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %foo : !firrtl.uint, out %bar: !firrtl.uint) {
  // CHECK: %[[const:.*]] = firrtl.constant 1
  // CHECK: firrtl.reg %clock
  // CHECK-NEXT:  firrtl.strictconnect %bar, %d : !firrtl.uint, !firrtl.uint
  // CHECK-NEXT:  %0 = firrtl.mux(%reset, %[[const]], %foo) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  // CHECK-NEXT:  firrtl.strictconnect %d, %0 : !firrtl.uint, !firrtl.uint
  // CHECK-NEXT: }
  %d = firrtl.reg %clock  : !firrtl.uint
  firrtl.strictconnect %bar, %d : !firrtl.uint, !firrtl.uint
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint
  %1 = firrtl.mux(%reset, %c1_ui2, %foo) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  firrtl.strictconnect %d, %1 : !firrtl.uint, !firrtl.uint
}

// CHECK-LABEL: @ComparisonOfUnsizedAndSized
firrtl.module @ComparisonOfUnsizedAndSized(
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>,
  out %y6: !firrtl.uint<1>,
  out %y7: !firrtl.uint<1>,
  out %y8: !firrtl.uint<1>,
  out %y9: !firrtl.uint<1>,
  out %y10: !firrtl.uint<1>,
  out %y11: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  %c3_si = firrtl.constant 3 : !firrtl.sint
  %c4_si4 = firrtl.constant 4 : !firrtl.sint<4>
  %c3_ui = firrtl.constant 3 : !firrtl.uint
  %c4_ui3 = firrtl.constant 4 : !firrtl.uint<3>

  %0 = firrtl.leq %c3_ui, %c4_ui3 : (!firrtl.uint, !firrtl.uint<3>) -> !firrtl.uint<1>
  %1 = firrtl.leq %c3_si, %c4_si4 : (!firrtl.sint, !firrtl.sint<4>) -> !firrtl.uint<1>
  %2 = firrtl.lt %c3_ui, %c4_ui3 : (!firrtl.uint, !firrtl.uint<3>) -> !firrtl.uint<1>
  %3 = firrtl.lt %c3_si, %c4_si4 : (!firrtl.sint, !firrtl.sint<4>) -> !firrtl.uint<1>
  %4 = firrtl.geq %c3_ui, %c4_ui3 : (!firrtl.uint, !firrtl.uint<3>) -> !firrtl.uint<1>
  %5 = firrtl.geq %c3_si, %c4_si4 : (!firrtl.sint, !firrtl.sint<4>) -> !firrtl.uint<1>
  %6 = firrtl.gt %c3_ui, %c4_ui3 : (!firrtl.uint, !firrtl.uint<3>) -> !firrtl.uint<1>
  %7 = firrtl.gt %c3_si, %c4_si4 : (!firrtl.sint, !firrtl.sint<4>) -> !firrtl.uint<1>
  %8 = firrtl.eq %c3_ui, %c4_ui3 : (!firrtl.uint, !firrtl.uint<3>) -> !firrtl.uint<1>
  %9 = firrtl.eq %c3_si, %c4_si4 : (!firrtl.sint, !firrtl.sint<4>) -> !firrtl.uint<1>
  %10 = firrtl.neq %c3_ui, %c4_ui3 : (!firrtl.uint, !firrtl.uint<3>) -> !firrtl.uint<1>
  %11 = firrtl.neq %c3_si, %c4_si4 : (!firrtl.sint, !firrtl.sint<4>) -> !firrtl.uint<1>

  firrtl.strictconnect %y0, %0 : !firrtl.uint<1>
  firrtl.strictconnect %y1, %1 : !firrtl.uint<1>
  firrtl.strictconnect %y2, %2 : !firrtl.uint<1>
  firrtl.strictconnect %y3, %3 : !firrtl.uint<1>
  firrtl.strictconnect %y4, %4 : !firrtl.uint<1>
  firrtl.strictconnect %y5, %5 : !firrtl.uint<1>
  firrtl.strictconnect %y6, %6 : !firrtl.uint<1>
  firrtl.strictconnect %y7, %7 : !firrtl.uint<1>
  firrtl.strictconnect %y8, %8 : !firrtl.uint<1>
  firrtl.strictconnect %y9, %9 : !firrtl.uint<1>
  firrtl.strictconnect %y10, %10 : !firrtl.uint<1>
  firrtl.strictconnect %y11, %11 : !firrtl.uint<1>
  // CHECK-NEXT: firrtl.strictconnect %y0, %c1_ui1
  // CHECK-NEXT: firrtl.strictconnect %y1, %c1_ui1
  // CHECK-NEXT: firrtl.strictconnect %y2, %c1_ui1
  // CHECK-NEXT: firrtl.strictconnect %y3, %c1_ui1
  // CHECK-NEXT: firrtl.strictconnect %y4, %c0_ui1
  // CHECK-NEXT: firrtl.strictconnect %y5, %c0_ui1
  // CHECK-NEXT: firrtl.strictconnect %y6, %c0_ui1
  // CHECK-NEXT: firrtl.strictconnect %y7, %c0_ui1
  // CHECK-NEXT: firrtl.strictconnect %y8, %c0_ui1
  // CHECK-NEXT: firrtl.strictconnect %y9, %c0_ui1
  // CHECK-NEXT: firrtl.strictconnect %y10, %c1_ui1
  // CHECK-NEXT: firrtl.strictconnect %y11, %c1_ui1
}

// CHECK-LABEL: @ComparisonOfUnsized
firrtl.module @ComparisonOfUnsized(
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>,
  out %y6: !firrtl.uint<1>,
  out %y7: !firrtl.uint<1>,
  out %y8: !firrtl.uint<1>,
  out %y9: !firrtl.uint<1>,
  out %y10: !firrtl.uint<1>,
  out %y11: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  %c0_si = firrtl.constant 0 : !firrtl.sint
  %c4_si = firrtl.constant 4 : !firrtl.sint
  %c0_ui = firrtl.constant 0 : !firrtl.uint
  %c4_ui = firrtl.constant 4 : !firrtl.uint

  %0 = firrtl.leq %c0_ui, %c4_ui : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  %1 = firrtl.leq %c0_si, %c4_si : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint<1>
  %2 = firrtl.lt %c0_ui, %c4_ui : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  %3 = firrtl.lt %c0_si, %c4_si : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint<1>
  %4 = firrtl.geq %c0_ui, %c4_ui : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  %5 = firrtl.geq %c0_si, %c4_si : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint<1>
  %6 = firrtl.gt %c0_ui, %c4_ui : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  %7 = firrtl.gt %c0_si, %c4_si : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint<1>
  %8 = firrtl.eq %c0_ui, %c4_ui : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  %9 = firrtl.eq %c0_si, %c4_si : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint<1>
  %10 = firrtl.neq %c0_ui, %c4_ui : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  %11 = firrtl.neq %c0_si, %c4_si : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint<1>

  firrtl.strictconnect %y0, %0 : !firrtl.uint<1>
  firrtl.strictconnect %y1, %1 : !firrtl.uint<1>
  firrtl.strictconnect %y2, %2 : !firrtl.uint<1>
  firrtl.strictconnect %y3, %3 : !firrtl.uint<1>
  firrtl.strictconnect %y4, %4 : !firrtl.uint<1>
  firrtl.strictconnect %y5, %5 : !firrtl.uint<1>
  firrtl.strictconnect %y6, %6 : !firrtl.uint<1>
  firrtl.strictconnect %y7, %7 : !firrtl.uint<1>
  firrtl.strictconnect %y8, %8 : !firrtl.uint<1>
  firrtl.strictconnect %y9, %9 : !firrtl.uint<1>
  firrtl.strictconnect %y10, %10 : !firrtl.uint<1>
  firrtl.strictconnect %y11, %11 : !firrtl.uint<1>
  // CHECK-NEXT: firrtl.strictconnect %y0, %c1_ui1
  // CHECK-NEXT: firrtl.strictconnect %y1, %c1_ui1
  // CHECK-NEXT: firrtl.strictconnect %y2, %c1_ui1
  // CHECK-NEXT: firrtl.strictconnect %y3, %c1_ui1
  // CHECK-NEXT: firrtl.strictconnect %y4, %c0_ui1
  // CHECK-NEXT: firrtl.strictconnect %y5, %c0_ui1
  // CHECK-NEXT: firrtl.strictconnect %y6, %c0_ui1
  // CHECK-NEXT: firrtl.strictconnect %y7, %c0_ui1
  // CHECK-NEXT: firrtl.strictconnect %y8, %c0_ui1
  // CHECK-NEXT: firrtl.strictconnect %y9, %c0_ui1
  // CHECK-NEXT: firrtl.strictconnect %y10, %c1_ui1
  // CHECK-NEXT: firrtl.strictconnect %y11, %c1_ui1
}


// CHECK-LABEL: firrtl.module @And
firrtl.module @And(in %in: !firrtl.uint<4>,
                   in %sin: !firrtl.sint<4>,
                   in %zin1: !firrtl.uint<0>,
                   in %zin2: !firrtl.uint<0>,
                   out %out: !firrtl.uint<4>,
                   out %outz: !firrtl.uint<0>) {
  // CHECK: firrtl.strictconnect %out, %c1_ui4
  %c1_ui4 = firrtl.constant 1 : !firrtl.uint<4>
  %c3_ui4 = firrtl.constant 3 : !firrtl.uint<4>
  %0 = firrtl.and %c3_ui4, %c1_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %0 : !firrtl.uint<4>

  // CHECK: firrtl.strictconnect %out, %in
  %c15_ui4 = firrtl.constant 15 : !firrtl.uint<4>
  %1 = firrtl.and %in, %c15_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %1 : !firrtl.uint<4>

  // CHECK: firrtl.strictconnect %out, %c0_ui4
  %c1_ui0 = firrtl.constant 0 : !firrtl.uint<4>
  %2 = firrtl.and %in, %c1_ui0 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %2 : !firrtl.uint<4>

  // CHECK: firrtl.strictconnect %out, %c0_ui4
  %inv_2 = firrtl.and %c1_ui0, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %inv_2 : !firrtl.uint<4>

  // CHECK: firrtl.strictconnect %out, %in
  %3 = firrtl.and %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %3 : !firrtl.uint<4>

  // CHECK: firrtl.strictconnect %out, %c0_ui4
  // CHECK: firrtl.strictconnect %outz, %c0_ui0
  %zw = firrtl.and %zin1, %zin2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  firrtl.connect %out, %zw : !firrtl.uint<4>, !firrtl.uint<0>
  firrtl.strictconnect %outz, %zw : !firrtl.uint<0>

  // Mixed type inputs - the constant is zero extended, not sign extended, so it
  // cannot be folded!

  // CHECK: firrtl.and %in, %c3_ui4
  // CHECK-NEXT: firrtl.strictconnect %out,
  %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>
  %4 = firrtl.and %in, %c3_ui2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %4 : !firrtl.uint<4>

  // Mixed type input and outputs.

  // CHECK: firrtl.strictconnect %out, %c1_ui4
  %c1_si4 = firrtl.constant 1 : !firrtl.sint<4>
  %5 = firrtl.and %c1_si4, %c1_si4 : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %5 : !firrtl.uint<4>

  // CHECK: %[[AND:.+]] = firrtl.asUInt %sin
  // CHECK-NEXT: firrtl.strictconnect %out, %[[AND]]
  %6 = firrtl.and %sin, %sin : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %6 : !firrtl.uint<4>

  // CHECK: firrtl.strictconnect %out, %c0_ui4
  %c0_si2 = firrtl.constant 0 : !firrtl.sint<2>
  %7 = firrtl.and %sin, %c0_si2 : (!firrtl.sint<4>, !firrtl.sint<2>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %7 : !firrtl.uint<4>

}


// CHECK-LABEL: firrtl.module @Or
firrtl.module @Or(in %in: !firrtl.uint<4>,
                  in %sin: !firrtl.sint<4>,
                  in %zin1: !firrtl.uint<0>,
                  in %zin2: !firrtl.uint<0>,
                  out %out: !firrtl.uint<4>,
                  out %outz: !firrtl.uint<0>) {
  // CHECK: firrtl.strictconnect %out, %c7_ui4
  %c4_ui4 = firrtl.constant 4 : !firrtl.uint<4>
  %c3_ui4 = firrtl.constant 3 : !firrtl.uint<4>
  %0 = firrtl.or %c3_ui4, %c4_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %0 : !firrtl.uint<4>

  // CHECK: firrtl.strictconnect %out, %c15_ui4
  %c1_ui15 = firrtl.constant 15 : !firrtl.uint<4>
  %1 = firrtl.or %in, %c1_ui15 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %1 : !firrtl.uint<4>

  // CHECK: firrtl.strictconnect %out, %in
  %c1_ui0 = firrtl.constant 0 : !firrtl.uint<4>
  %2 = firrtl.or %in, %c1_ui0 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %2 : !firrtl.uint<4>

  // CHECK: firrtl.strictconnect %out, %in
  %inv_2 = firrtl.or %c1_ui0, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %inv_2 : !firrtl.uint<4>

  // CHECK: firrtl.strictconnect %out, %in
  %3 = firrtl.or %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %3 : !firrtl.uint<4>

  // CHECK: firrtl.strictconnect %out, %c0_ui4
  // CHECK: firrtl.strictconnect %outz, %c0_ui0
  %zw = firrtl.or %zin1, %zin2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  firrtl.connect %out, %zw : !firrtl.uint<4>, !firrtl.uint<0>
  firrtl.strictconnect %outz, %zw : !firrtl.uint<0>

  // Mixed type input and outputs.

  // CHECK: firrtl.strictconnect %out, %c1_ui4
  %c1_si4 = firrtl.constant 1 : !firrtl.sint<4>
  %5 = firrtl.or %c1_si4, %c1_si4 : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %5 : !firrtl.uint<4>

  // CHECK: [[OR:%.+]] = firrtl.asUInt %sin
  // CHECK-NEXT: firrtl.strictconnect %out, [[OR]]
  %6 = firrtl.or %sin, %sin : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %6 : !firrtl.uint<4>

  // CHECK: firrtl.strictconnect %out, %c15_ui4
  %c0_si2 = firrtl.constant -1 : !firrtl.sint<2>
  %7 = firrtl.or %sin, %c0_si2 : (!firrtl.sint<4>, !firrtl.sint<2>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %7 : !firrtl.uint<4>
}


// CHECK-LABEL: firrtl.module @Xor
firrtl.module @Xor(in %in: !firrtl.uint<4>,
                   in %sin: !firrtl.sint<4>,
                   in %zin1: !firrtl.uint<0>,
                   in %zin2: !firrtl.uint<0>,
                   out %out: !firrtl.uint<4>,
                   out %outz: !firrtl.uint<0>) {
  // CHECK: firrtl.strictconnect %out, %c2_ui4
  %c1_ui4 = firrtl.constant 1 : !firrtl.uint<4>
  %c3_ui4 = firrtl.constant 3 : !firrtl.uint<4>
  %0 = firrtl.xor %c3_ui4, %c1_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %0 : !firrtl.uint<4>

  // CHECK: firrtl.strictconnect %out, %in
  %c1_ui0 = firrtl.constant 0 : !firrtl.uint<4>
  %2 = firrtl.xor %in, %c1_ui0 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %2 : !firrtl.uint<4>

  // CHECK: firrtl.strictconnect %out, %c0_ui4
  %3 = firrtl.xor %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %3 : !firrtl.uint<4>

  // CHECK: firrtl.strictconnect %out, %c0_ui4
  // CHECK: firrtl.strictconnect %outz, %c0_ui0
  %zw = firrtl.xor %zin1, %zin2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  firrtl.connect %out, %zw : !firrtl.uint<4>, !firrtl.uint<0>
  firrtl.strictconnect %outz, %zw : !firrtl.uint<0>

  // Mixed type input and outputs.

  // CHECK: firrtl.strictconnect %out, %c0_ui4
  %6 = firrtl.xor %sin, %sin : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %6 : !firrtl.uint<4>

  // CHECK: %[[aui:.*]] = firrtl.asUInt %sin
  // CHECK: firrtl.strictconnect %out, %[[aui]]
  %c0_si2 = firrtl.constant 0 : !firrtl.sint<2>
  %7 = firrtl.xor %sin, %c0_si2 : (!firrtl.sint<4>, !firrtl.sint<2>) -> !firrtl.uint<4>
  firrtl.strictconnect %out, %7 : !firrtl.uint<4>
}



  // Issue #1064: https://github.com/llvm/circt/issues/1064
  // CHECK: firrtl.strictconnect %out1u, %c0_ui1
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %9 = firrtl.dshr %in0u, %c1_ui1 : (!firrtl.uint<0>, !firrtl.uint<1>) -> !firrtl.uint<0>
  firrtl.connect %out1u, %9 : !firrtl.uint<1>, !firrtl.uint<0>


// CHECK-LABEL: firrtl.module @Issue1188
// https://github.com/llvm/circt/issues/1188
// Make sure that we handle recursion through muxes correctly.
firrtl.circuit "Issue1188"  {
  firrtl.module @Issue1188(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %io_out: !firrtl.uint<6>, out %io_out3: !firrtl.uint<3>) {
    %c1_ui6 = firrtl.constant 1 : !firrtl.uint<6>
    %D0123456 = firrtl.reg %clock  : !firrtl.uint<6>
    %0 = firrtl.bits %D0123456 4 to 0 : (!firrtl.uint<6>) -> !firrtl.uint<5>
    %1 = firrtl.bits %D0123456 5 to 5 : (!firrtl.uint<6>) -> !firrtl.uint<1>
    %2 = firrtl.cat %0, %1 : (!firrtl.uint<5>, !firrtl.uint<1>) -> !firrtl.uint<6>
    %3 = firrtl.bits %D0123456 4 to 4 : (!firrtl.uint<6>) -> !firrtl.uint<1>
    %4 = firrtl.xor %2, %3 : (!firrtl.uint<6>, !firrtl.uint<1>) -> !firrtl.uint<6>
    %5 = firrtl.bits %D0123456 1 to 1 : (!firrtl.uint<6>) -> !firrtl.uint<1>
    %6 = firrtl.bits %D0123456 3 to 3 : (!firrtl.uint<6>) -> !firrtl.uint<1>
    %7 = firrtl.cat %5, %6 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    %8 = firrtl.cat %7, %1 : (!firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<3>
    firrtl.strictconnect %io_out, %D0123456 : !firrtl.uint<6>
    firrtl.strictconnect %io_out3, %8 : !firrtl.uint<3>
    // CHECK: firrtl.mux(%reset, %c1_ui6, %4)
    %9 = firrtl.mux(%reset, %c1_ui6, %4) : (!firrtl.uint<1>, !firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
    firrtl.strictconnect %D0123456, %9 : !firrtl.uint<6>
  }
}

