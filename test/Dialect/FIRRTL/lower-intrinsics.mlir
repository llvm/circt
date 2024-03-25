// Test intmodule -> ops + LowerIntrinsics combined, to start with.
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-intmodules,firrtl.module(firrtl-lower-intrinsics)))' %s | FileCheck %s --check-prefixes=CHECK,CHECK-NOEICG
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-intmodules{fixup-eicg-wrapper},firrtl.module(firrtl-lower-intrinsics)))' %s | FileCheck %s --check-prefixes=CHECK,CHECK-EICG

// CHECK-LABEL: "Foo"
firrtl.circuit "Foo" {
  // CHECK-NOT: NameDoesNotMatter5
  firrtl.intmodule @NameDoesNotMatter5(in i : !firrtl.clock, out size : !firrtl.uint<32>) attributes
                                     {intrinsic = "circt.sizeof"}
  // CHECK-NOT: NameDoesNotMatter6
  firrtl.intmodule @NameDoesNotMatter6(in i : !firrtl.clock, out found : !firrtl.uint<1>) attributes
                                     {intrinsic = "circt.isX"}
  // CHECK-NOT: NameDoesNotMatter7
  firrtl.intmodule @NameDoesNotMatter7<FORMAT: none = "foo">(out found : !firrtl.uint<1>) attributes
                                     {intrinsic = "circt.plusargs.test"}
  // CHECK-NOT: NameDoesNotMatter8
  firrtl.intmodule @NameDoesNotMatter8<FORMAT: none = "foo">(out found : !firrtl.uint<1>, out result: !firrtl.uint<5>) attributes
                                     {intrinsic = "circt.plusargs.value"}

  // CHECK: Foo
  firrtl.module @Foo(in %clk : !firrtl.clock, out %s : !firrtl.uint<32>, out %io1 : !firrtl.uint<1>, out %io2 : !firrtl.uint<1>, out %io3 : !firrtl.uint<1>, out %io4 : !firrtl.uint<5>) {
    %i1, %size = firrtl.instance "" @NameDoesNotMatter5(in i : !firrtl.clock, out size : !firrtl.uint<32>)
    // CHECK-NOT: NameDoesNotMatter5
    // CHECK: firrtl.int.sizeof
    firrtl.strictconnect %i1, %clk : !firrtl.clock
    firrtl.strictconnect %s, %size : !firrtl.uint<32>

    %i2, %found2 = firrtl.instance "" @NameDoesNotMatter6(in i : !firrtl.clock, out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter6
    // CHECK: firrtl.int.isX
    firrtl.strictconnect %i2, %clk : !firrtl.clock
    firrtl.strictconnect %io1, %found2 : !firrtl.uint<1>

    %found3 = firrtl.instance "" @NameDoesNotMatter7(out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter7
    // CHECK: firrtl.int.plusargs.test "foo"
    firrtl.strictconnect %io2, %found3 : !firrtl.uint<1>

    %found4, %result1 = firrtl.instance "" @NameDoesNotMatter8(out found : !firrtl.uint<1>, out result: !firrtl.uint<5>)
    // CHECK-NOT: NameDoesNotMatter8
    // CHECK: firrtl.int.plusargs.value "foo" : !firrtl.uint<5>
    firrtl.strictconnect %io3, %found4 : !firrtl.uint<1>
    firrtl.strictconnect %io4, %result1 : !firrtl.uint<5>
  }

  // CHECK-NOT: ClockGate1
  firrtl.intmodule @ClockGate1(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {intrinsic = "circt.clock_gate"}

  // CHECK: ClockGate
  firrtl.module @ClockGate(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>) {
    // CHECK-NOT: ClockGate1
    // CHECK: firrtl.int.clock_gate
    %in2, %en2, %out2 = firrtl.instance "" @ClockGate1(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.strictconnect %in2, %clk : !firrtl.clock
    firrtl.strictconnect %en2, %en : !firrtl.uint<1>
  }

  // CHECK-NOT: ClockInverter1
  firrtl.intmodule @ClockInverter1(in in: !firrtl.clock, out out: !firrtl.clock) attributes {intrinsic = "circt.clock_inv"}

  // CHECK: ClockInverter
  firrtl.module @ClockInverter(in %clk: !firrtl.clock) {
    // CHECK-NOT: ClockInverter1
    // CHECK: firrtl.int.clock_inv
    %in2, %out2 = firrtl.instance "" @ClockInverter1(in in: !firrtl.clock, out out: !firrtl.clock)
    firrtl.strictconnect %in2, %clk : !firrtl.clock
  }

  // CHECK-NOT: LTLAnd
  // CHECK-NOT: LTLOr
  // CHECK-NOT: LTLDelay1
  // CHECK-NOT: LTLDelay2
  // CHECK-NOT: LTLConcat
  // CHECK-NOT: LTLNot
  // CHECK-NOT: LTLImplication
  // CHECK-NOT: LTLEventually
  // CHECK-NOT: LTLClock
  // CHECK-NOT: LTLDisable
  firrtl.intmodule @LTLAnd(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.and"}
  firrtl.intmodule @LTLOr(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.or"}
  firrtl.intmodule @LTLDelay1<delay: i64 = 42>(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.delay"}
  firrtl.intmodule @LTLDelay2<delay: i64 = 42, length: i64 = 1337>(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.delay"}
  firrtl.intmodule @LTLConcat(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.concat"}
  firrtl.intmodule @LTLNot(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.not"}
  firrtl.intmodule @LTLImplication(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.implication"}
  firrtl.intmodule @LTLEventually(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.eventually"}
  firrtl.intmodule @LTLClock(in in: !firrtl.uint<1>, in clock: !firrtl.clock, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.clock"}
  firrtl.intmodule @LTLDisable(in in: !firrtl.uint<1>, in condition: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.disable"}

  // CHECK: firrtl.module @LTL()
  firrtl.module @LTL() {
    // CHECK-NOT: LTLAnd
    // CHECK-NOT: LTLOr
    // CHECK: firrtl.int.ltl.and {{%.+}}, {{%.+}} :
    // CHECK: firrtl.int.ltl.or {{%.+}}, {{%.+}} :
    %and.lhs, %and.rhs, %and.out = firrtl.instance "and" @LTLAnd(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %or.lhs, %or.rhs, %or.out = firrtl.instance "or" @LTLOr(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>)

    // CHECK-NOT: LTLDelay1
    // CHECK-NOT: LTLDelay2
    // CHECK: firrtl.int.ltl.delay {{%.+}}, 42 :
    // CHECK: firrtl.int.ltl.delay {{%.+}}, 42, 1337 :
    %delay1.in, %delay1.out = firrtl.instance "delay1" @LTLDelay1(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %delay2.in, %delay2.out = firrtl.instance "delay2" @LTLDelay2(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)

    // CHECK-NOT: LTLConcat
    // CHECK-NOT: LTLNot
    // CHECK-NOT: LTLImplication
    // CHECK-NOT: LTLEventually
    // CHECK: firrtl.int.ltl.concat {{%.+}}, {{%.+}} :
    // CHECK: firrtl.int.ltl.not {{%.+}} :
    // CHECK: firrtl.int.ltl.implication {{%.+}}, {{%.+}} :
    // CHECK: firrtl.int.ltl.eventually {{%.+}} :
    %concat.lhs, %concat.rhs, %concat.out = firrtl.instance "concat" @LTLConcat(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %not.in, %not.out = firrtl.instance "not" @LTLNot(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %implication.lhs, %implication.rhs, %implication.out = firrtl.instance "implication" @LTLImplication(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %eventually.in, %eventually.out = firrtl.instance "eventually" @LTLEventually(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)

    // CHECK-NOT: LTLClock
    // CHECK: firrtl.int.ltl.clock {{%.+}}, {{%.+}} :
    %clock.in, %clock.clock, %clock.out = firrtl.instance "clock" @LTLClock(in in: !firrtl.uint<1>, in clock: !firrtl.clock, out out: !firrtl.uint<1>)

    // CHECK-NOT: LTLDisable
    // CHECK: firrtl.int.ltl.disable {{%.+}}, {{%.+}} :
    %disable.in, %disable.condition, %disable.out = firrtl.instance "disable" @LTLDisable(in in: !firrtl.uint<1>, in condition: !firrtl.uint<1>, out out: !firrtl.uint<1>)
  }

  // CHECK-NOT: VerifAssert1
  // CHECK-NOT: VerifAssert2
  // CHECK-NOT: VerifAssume
  // CHECK-NOT: VerifCover
  firrtl.intmodule @VerifAssert1(in property: !firrtl.uint<1>) attributes {intrinsic = "circt.verif.assert"}
  firrtl.intmodule @VerifAssert2<label: none = "hello">(in property: !firrtl.uint<1>) attributes {intrinsic = "circt.verif.assert"}
  firrtl.intmodule @VerifAssume(in property: !firrtl.uint<1>) attributes {intrinsic = "circt.verif.assume"}
  firrtl.intmodule @VerifCover(in property: !firrtl.uint<1>) attributes {intrinsic = "circt.verif.cover"}

  // CHECK: firrtl.module @Verif()
  firrtl.module @Verif() {
    // CHECK-NOT: VerifAssert1
    // CHECK-NOT: VerifAssert2
    // CHECK-NOT: VerifAssume
    // CHECK-NOT: VerifCover
    // CHECK: firrtl.int.verif.assert {{%.+}} :
    // CHECK: firrtl.int.verif.assert {{%.+}} {label = "hello"} :
    // CHECK: firrtl.int.verif.assume {{%.+}} :
    // CHECK: firrtl.int.verif.cover {{%.+}} :
    %assert1.property = firrtl.instance "assert1" @VerifAssert1(in property: !firrtl.uint<1>)
    %assert2.property = firrtl.instance "assert2" @VerifAssert2(in property: !firrtl.uint<1>)
    %assume.property = firrtl.instance "assume" @VerifAssume(in property: !firrtl.uint<1>)
    %cover.property = firrtl.instance "cover" @VerifCover(in property: !firrtl.uint<1>)
  }

  firrtl.intmodule @Mux2Cell(in sel: !firrtl.uint<1>, in high: !firrtl.uint, in low: !firrtl.uint, out out: !firrtl.uint) attributes {intrinsic = "circt.mux2cell"}
  firrtl.intmodule @Mux4Cell(in sel: !firrtl.uint<2>, in v3: !firrtl.uint, in v2: !firrtl.uint, in v1: !firrtl.uint, in v0: !firrtl.uint, out out: !firrtl.uint) attributes {intrinsic = "circt.mux4cell"}

  // CHECK: firrtl.module @MuxCell()
  firrtl.module @MuxCell() {
    // CHECK: firrtl.int.mux2cell
    // CHECK: firrtl.int.mux4cell
    %sel_0, %high, %low, %out_0 = firrtl.instance "mux2" @Mux2Cell(in sel: !firrtl.uint<1>, in high: !firrtl.uint, in low: !firrtl.uint, out out: !firrtl.uint)
    %sel_1, %v4, %v3, %v2, %v1, %out_1 = firrtl.instance "mux4" @Mux4Cell(in sel: !firrtl.uint<2>, in v3: !firrtl.uint, in v2: !firrtl.uint, in v1: !firrtl.uint, in v0: !firrtl.uint, out out: !firrtl.uint)
  }

  // CHECK-NOT: HBRInt1
  // CHECK-NOT: HBRInt2
  // CHECK-NOT: HBRInt3
  firrtl.intmodule @HBRInt1(in clock: !firrtl.clock, in reset: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.has_been_reset"}
  firrtl.intmodule @HBRInt2(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.has_been_reset"}
  firrtl.intmodule @HBRInt3(in clock: !firrtl.clock, in reset: !firrtl.reset, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.has_been_reset"}

  // CHECK: HasBeenReset
  firrtl.module @HasBeenReset(in %clock: !firrtl.clock, in %reset1: !firrtl.uint<1>, in %reset2: !firrtl.asyncreset, in %reset3: !firrtl.reset) {
    // CHECK-NOT: firrtl.instance
    // CHECK: firrtl.int.has_been_reset {{%.+}}, {{%.+}} : !firrtl.uint<1>
    // CHECK-NOT: firrtl.instance
    // CHECK: firrtl.int.has_been_reset {{%.+}}, {{%.+}} : !firrtl.asyncreset
    // CHECK-NOT: firrtl.instance
    // CHECK: firrtl.int.has_been_reset {{%.+}}, {{%.+}} : !firrtl.reset
    %in_clock1, %in_reset1, %hbr1 = firrtl.instance "" @HBRInt1(in clock: !firrtl.clock, in reset: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %in_clock2, %in_reset2, %hbr2 = firrtl.instance "" @HBRInt2(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, out out: !firrtl.uint<1>)
    %in_clock3, %in_reset3, %hbr3 = firrtl.instance "" @HBRInt3(in clock: !firrtl.clock, in reset: !firrtl.reset, out out: !firrtl.uint<1>)
    firrtl.strictconnect %in_clock1, %clock : !firrtl.clock
    firrtl.strictconnect %in_clock2, %clock : !firrtl.clock
    firrtl.strictconnect %in_clock3, %clock : !firrtl.clock
    firrtl.strictconnect %in_reset1, %reset1 : !firrtl.uint<1>
    firrtl.strictconnect %in_reset2, %reset2 : !firrtl.asyncreset
    firrtl.strictconnect %in_reset3, %reset3 : !firrtl.reset
  }

  // CHECK-NOEICG: LegacyClockGate
  // CHECK-EICG-NOT: LegacyClockGate
  firrtl.extmodule @LegacyClockGate(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}

  // CHECK: FixupEICGWrapper
  firrtl.module @FixupEICGWrapper(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) {
    // CHECK-NOEICG: firrtl.instance
    // CHECK-EICG-NOT: firrtl.instance
    // CHECK-EICG: firrtl.int.clock_gate
    %ckg_in, %ckg_test_en, %ckg_en, %ckg_out = firrtl.instance ckg @LegacyClockGate(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.strictconnect %ckg_in, %clock : !firrtl.clock
    firrtl.strictconnect %ckg_test_en, %en : !firrtl.uint<1>
    firrtl.strictconnect %ckg_en, %en : !firrtl.uint<1>
  }

  // CHECK-NOT: CirctAssert1
  // CHECK-NOT: CirctAssert2
  // CHECK-NOT: VerifAssume
  // CHECK-NOT: VerifCover
// TODO:
  firrtl.intmodule private @AssertAssume<format: none = "testing">(in clock: !firrtl.clock, in predicate: !firrtl.uint<1>, in enable: !firrtl.uint<1>) attributes {intrinsic = "circt.chisel_assert_assume"}
  firrtl.intmodule private @AssertAssumeFormat<format: none = "message: %d",
                                               label: none = "label for assert with format string",
                                               guards: none = "MACRO_GUARD;ASDF">(
                                                 in clock: !firrtl.clock,
                                                 in predicate: !firrtl.uint<1>,
                                                 in enable: !firrtl.uint<1>,
                                                 in val: !firrtl.uint<1>
                                               ) attributes {intrinsic = "circt.chisel_assert_assume"}
  firrtl.intmodule private @IfElseFatalFormat<format: none = "ief: %d",
                                              label: none = "label for ifelsefatal assert",
                                              guards: none = "MACRO_GUARD;ASDF">(
                                                in clock: !firrtl.clock,
                                                in predicate: !firrtl.uint<1>,
                                                in enable: !firrtl.uint<1>,
                                                in val: !firrtl.uint<1>
                                              ) attributes {intrinsic = "circt.chisel_ifelsefatal"}
  firrtl.intmodule private @Assume<format: none = "text: %d",
                                   label: none = "label for assume">(
                                     in clock: !firrtl.clock,
                                     in predicate: !firrtl.uint<1>,
                                     in enable: !firrtl.uint<1>,
                                     in val: !firrtl.uint<1>
                                   ) attributes {intrinsic = "circt.chisel_assume"}
  firrtl.intmodule private @CoverLabel<label: none = "label for cover">(
                                         in clock: !firrtl.clock,
                                         in predicate: !firrtl.uint<1>,
                                         in enable: !firrtl.uint<1>
                                       ) attributes {intrinsic = "circt.chisel_cover"}
  // CHECK-NOT: @AssertAssume
  // CHECK-NOT: @AssertAssumeFormat
  // CHECK-NOT: @IfElseFatalFormat
  // CHECK-NOT: @Assume
  // CHECK-NOT: @CoverLabel

  // CHECK: firrtl.module @ChiselVerif(
  firrtl.module @ChiselVerif(in %clock: !firrtl.clock,
                             in %cond: !firrtl.uint<1>,
                             in %enable: !firrtl.uint<1>) {
    // CHECK-NOT: firrtl.instance
    // CHECK: firrtl.assert %{{.+}}, %{{.+}}, %{{.+}}, "testing" :
    // CHECK-SAME: isConcurrent = true
    %assert_clock, %assert_predicate, %assert_enable = firrtl.instance assert interesting_name @AssertAssume(in clock: !firrtl.clock, in predicate: !firrtl.uint<1>, in enable: !firrtl.uint<1>)
    firrtl.strictconnect %assert_clock, %clock : !firrtl.clock
    firrtl.strictconnect %assert_predicate, %cond : !firrtl.uint<1>
    firrtl.strictconnect %assert_enable, %enable : !firrtl.uint<1>
    // CHECK-NOT: firrtl.instance
    // CHECK: firrtl.assert %{{.+}}, %{{.+}}, %{{.+}}, "message: %d"(
    // CHECK-SAME: guards = ["MACRO_GUARD", "ASDF"]
    // CHECK-SAME: isConcurrent = true
    // CHECK-SAME: name = "label for assert with format string"
    %assertFormat_clock, %assertFormat_predicate, %assertFormat_enable, %assertFormat_val = firrtl.instance assertFormat interesting_name @AssertAssumeFormat(in clock: !firrtl.clock, in predicate: !firrtl.uint<1>, in enable: !firrtl.uint<1>, in val: !firrtl.uint<1>)
    firrtl.strictconnect %assertFormat_clock, %clock : !firrtl.clock
    firrtl.strictconnect %assertFormat_predicate, %cond : !firrtl.uint<1>
    firrtl.strictconnect %assertFormat_enable, %enable : !firrtl.uint<1>
    firrtl.strictconnect %assertFormat_val, %cond : !firrtl.uint<1>
    // CHECK-NOT: firrtl.instance
    // CHECK: firrtl.assert %{{.+}}, %{{.+}}, %{{.+}}, "ief: %d"(
    // CHECK-SAME: format = "ifElseFatal"
    // CHECK-SAME: guards = ["MACRO_GUARD", "ASDF"]
    // CHECK-SAME: isConcurrent = true
    // CHECK-SAME: name = "label for ifelsefatal assert"
    %ief_clock, %ief_predicate, %ief_enable, %ief_val = firrtl.instance ief interesting_name @IfElseFatalFormat(in clock: !firrtl.clock, in predicate: !firrtl.uint<1>, in enable: !firrtl.uint<1>, in val: !firrtl.uint<1>)
    firrtl.strictconnect %ief_clock, %clock : !firrtl.clock
    firrtl.strictconnect %ief_predicate, %cond : !firrtl.uint<1>
    firrtl.strictconnect %ief_enable, %enable : !firrtl.uint<1>
    firrtl.strictconnect %ief_val, %enable : !firrtl.uint<1>
    // CHECK-NOT: firrtl.instance
    // CHECK: firrtl.assume %{{.+}}, %{{.+}}, %{{.+}}, "text: %d"(
    // CHECK-SAME: isConcurrent = true
    // CHECK-SAME: name = "label for assume"
    %assume_clock, %assume_predicate, %assume_enable, %assume_val = firrtl.instance assume interesting_name @Assume(in clock: !firrtl.clock, in predicate: !firrtl.uint<1>, in enable: !firrtl.uint<1>, in val: !firrtl.uint<1>)
    firrtl.strictconnect %assume_clock, %clock : !firrtl.clock
    firrtl.strictconnect %assume_predicate, %cond : !firrtl.uint<1>
    firrtl.strictconnect %assume_enable, %enable : !firrtl.uint<1>
    firrtl.strictconnect %assume_val, %enable : !firrtl.uint<1>
    // CHECK-NOT: firrtl.instance
    // CHECK: firrtl.cover %{{.+}}, %{{.+}}, %{{.+}}, "" :
    // CHECK-SAME: isConcurrent = true
    // CHECK-SAME: name = "label for cover"
    %cover_clock, %cover_predicate, %cover_enable = firrtl.instance cover interesting_name @CoverLabel(in clock: !firrtl.clock, in predicate: !firrtl.uint<1>, in enable: !firrtl.uint<1>)
    firrtl.strictconnect %cover_clock, %clock : !firrtl.clock
    firrtl.strictconnect %cover_predicate, %cond : !firrtl.uint<1>
    firrtl.strictconnect %cover_enable, %enable : !firrtl.uint<1>
  }

  // CHECK-NOT: firrtl.intmodule private @FPGAProbeIntrinsic
  firrtl.intmodule private @FPGAProbeIntrinsic(in data: !firrtl.uint, in clock: !firrtl.clock) attributes {intrinsic = "circt_fpga_probe"}

  // CHECK-LABEL: firrtl.module private @ProbeIntrinsicTest
  firrtl.module private @ProbeIntrinsicTest(in %clock : !firrtl.clock, in %data : !firrtl.uint<32>) {
    // CHECK:      [[DATA:%.+]] = firrtl.wire : !firrtl.uint
    // CHECK-NEXT: [[CLOCK:%.+]] = firrtl.wire : !firrtl.clock
    // CHECK-NEXT: firrtl.int.fpga_probe [[CLOCK]], [[DATA]] : !firrtl.uint
    // CHECK-NEXT: firrtl.strictconnect [[CLOCK]], %clock : !firrtl.clock
    // CHECK-NEXT: firrtl.connect [[DATA]], %data : !firrtl.uint, !firrtl.uint<32>
    %mod_data, %mod_clock = firrtl.instance mod @FPGAProbeIntrinsic(in data: !firrtl.uint, in clock: !firrtl.clock)
    firrtl.strictconnect %mod_clock, %clock : !firrtl.clock
    firrtl.connect %mod_data, %data : !firrtl.uint, !firrtl.uint<32>
  }
}

// CHECK-LABEL: "FixupEICGWrapper2"
firrtl.circuit "FixupEICGWrapper2" {
  // CHECK-NOEICG: LegacyClockGateNoTestEn
  // CHECK-EICG-NOT: LegacyClockGateNoTestEn
  firrtl.extmodule @LegacyClockGateNoTestEn(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}

  // CHECK: FixupEICGWrapper2
  firrtl.module @FixupEICGWrapper2(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) {
    // CHECK-NOEICG: firrtl.instance
    // CHECK-EICG-NOT: firrtl.instance
    // CHECK-EICG: firrtl.int.clock_gate
    %ckg_in, %ckg_en, %ckg_out = firrtl.instance ckg @LegacyClockGateNoTestEn(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.strictconnect %ckg_in, %clock : !firrtl.clock
    firrtl.strictconnect %ckg_en, %en : !firrtl.uint<1>
  }
}
