// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-lower-intrinsics)))' %s | FileCheck %s --check-prefixes=CHECK --implicit-check-not firrtl.int.generic

// CHECK-LABEL: "Foo"
firrtl.circuit "Foo" {
  // CHECK-LABEL: @Foo
  firrtl.module @Foo(in %clk : !firrtl.clock, out %s : !firrtl.uint<32>, out %io1 : !firrtl.uint<1>, out %io2 : !firrtl.uint<1>, out %io3 : !firrtl.uint<1>, out %io4 : !firrtl.uint<5>) {
    // CHECK: firrtl.int.sizeof %clk
    %size = firrtl.int.generic "circt.sizeof"  %clk : (!firrtl.clock) -> !firrtl.uint<32>
    firrtl.matchingconnect %s, %size : !firrtl.uint<32>

    // CHECK: firrtl.int.isX
    %isX = firrtl.int.generic "circt.isX"  %clk : (!firrtl.clock) -> !firrtl.uint<1>
    firrtl.matchingconnect %io1, %isX : !firrtl.uint<1>

    // CHECK: firrtl.int.plusargs.test "foo"
    %foo = firrtl.int.generic "circt.plusargs.test" <FORMAT: none = "foo"> : () -> !firrtl.uint<1>
    firrtl.matchingconnect %io2, %foo : !firrtl.uint<1>

    // CHECK: firrtl.int.plusargs.value "foo" : !firrtl.uint<5>
    %pav = firrtl.int.generic "circt.plusargs.value" <FORMAT: none = "foo"> : () -> !firrtl.bundle<found: uint<1>, result: uint<5>>
    %found = firrtl.subfield %pav[found] : !firrtl.bundle<found: uint<1>, result: uint<5>>
    %result = firrtl.subfield %pav[result] : !firrtl.bundle<found: uint<1>, result: uint<5>>
    firrtl.matchingconnect %io3, %found : !firrtl.uint<1>
    firrtl.matchingconnect %io4, %result : !firrtl.uint<5>
  }
  // CHECK-LABEL: @ClockGate
  firrtl.module @ClockGate(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>) {
    // CHECK-NEXT: firrtl.int.clock_gate %clk, %en
    firrtl.int.generic "circt.clock_gate"  %clk, %en : (!firrtl.clock, !firrtl.uint<1>) -> !firrtl.clock
    // CHECK-NEXT: firrtl.int.clock_gate %clk, %en, %en
    firrtl.int.generic "circt.clock_gate"  %clk, %en, %en : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.clock
  }

  // CHECK-LABEL: @ClockInverter
  firrtl.module @ClockInverter(in %clk: !firrtl.clock) {
    // CHECK-NEXT: firrtl.int.clock_inv %clk
    firrtl.int.generic "circt.clock_inv"  %clk : (!firrtl.clock) -> !firrtl.clock
  }

  // CHECK-LABEL: @ClockDivider
  firrtl.module @ClockDivider(in %clk: !firrtl.clock) {
    // CHECK-NEXT: firrtl.int.clock_div %clk by 8
    firrtl.int.generic "circt.clock_div" <POW_2: i8 = 8> %clk : (!firrtl.clock) -> !firrtl.clock
  }

  // CHECK-LABEL: firrtl.module @LTL(
  firrtl.module @LTL(in %in0 : !firrtl.uint<1>, in %in1 : !firrtl.uint<1>, in %clk : !firrtl.clock) {
    // CHECK-NEXT: firrtl.int.ltl.and %in0, %in1 :
    firrtl.int.generic "circt_ltl_and"  %in0, %in1: (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: firrtl.int.ltl.or %in0, %in1 :
    firrtl.int.generic "circt_ltl_or"  %in0, %in1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: firrtl.int.ltl.intersect %in0, %in1 :
    firrtl.int.generic "circt_ltl_intersect"  %in0, %in1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>

    // CHECK-NEXT: firrtl.int.ltl.delay %in0, 42 :
    firrtl.int.generic "circt_ltl_delay" <delay: i64 = 42> %in0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: firrtl.int.ltl.delay %in0, 42, 1337 :
    firrtl.int.generic "circt_ltl_delay" <delay: i64 = 42, length: i64 = 1337> %in0 : (!firrtl.uint<1>) -> !firrtl.uint<1>

    // CHECK-NEXT: firrtl.int.ltl.repeat %in0, 42 :
    firrtl.int.generic "circt_ltl_repeat" <base: i64 = 42> %in0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: firrtl.int.ltl.repeat %in0, 42, 1337 :
    firrtl.int.generic "circt_ltl_repeat" <base: i64 = 42, more: i64 = 1337> %in0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: firrtl.int.ltl.goto_repeat %in0, 42, 1337 :
    firrtl.int.generic "circt_ltl_goto_repeat" <base: i64 = 42, more: i64 = 1337> %in0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: firrtl.int.ltl.non_consecutive_repeat %in0, 42, 1337 :
    firrtl.int.generic "circt_ltl_non_consecutive_repeat" <base: i64 = 42, more: i64 = 1337> %in0 : (!firrtl.uint<1>) -> !firrtl.uint<1>

    // CHECK-NEXT: firrtl.int.ltl.concat %in0, %in1 :
    firrtl.int.generic "circt_ltl_concat"  %in0, %in1: (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: firrtl.int.ltl.not %in0 :
    firrtl.int.generic "circt_ltl_not"  %in0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: firrtl.int.ltl.implication %in0, %in1 :
    firrtl.int.generic "circt_ltl_implication"  %in0, %in1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: firrtl.int.ltl.until %in0, %in1 :
    firrtl.int.generic "circt_ltl_until"  %in0, %in1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: firrtl.int.ltl.eventually %in0 :
    firrtl.int.generic "circt_ltl_eventually"  %in0 : (!firrtl.uint<1>) -> !firrtl.uint<1>

    // CHECK-NEXT: firrtl.int.ltl.clock %in0, %clk :
    firrtl.int.generic "circt_ltl_clock"  %in0, %clk : (!firrtl.uint<1>, !firrtl.clock) -> !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @Verif(
  firrtl.module @Verif(in %in : !firrtl.uint<1>) {
    // CHECK-NEXT: firrtl.int.verif.assert %in :
    // CHECK-NEXT: firrtl.int.verif.assert %in {label = "hello"} :
    // CHECK-NEXT: firrtl.int.verif.assume %in :
    // CHECK-NEXT: firrtl.int.verif.cover %in :
    // CHECK-NEXT: firrtl.int.verif.require %in :
    // CHECK-NEXT: firrtl.int.verif.ensure %in :
    firrtl.int.generic "circt_verif_assert"  %in : (!firrtl.uint<1>) -> ()
    firrtl.int.generic "circt_verif_assert" <label: none = "hello"> %in : (!firrtl.uint<1>) -> ()
    firrtl.int.generic "circt_verif_assume"  %in : (!firrtl.uint<1>) -> ()
    firrtl.int.generic "circt_verif_cover"  %in : (!firrtl.uint<1>) -> ()
    firrtl.int.generic "circt_verif_require"  %in : (!firrtl.uint<1>) -> ()
    firrtl.int.generic "circt_verif_ensure"  %in : (!firrtl.uint<1>) -> ()
  }

  // CHECK-LABEL: firrtl.module private @MuxCell(
  firrtl.module private @MuxCell(in %sel : !firrtl.uint<1>,
                                 in %sel2 : !firrtl.uint<2>,
                                 in %d1 : !firrtl.uint,
                                 in %d2 : !firrtl.uint,
                                 in %d3 : !firrtl.uint,
                                 in %d4 : !firrtl.uint) {
    // CHECK-NEXT: firrtl.int.mux2cell(%sel, %d1, %d2)
    // CHECK-NEXT: firrtl.int.mux4cell(%sel2, %d1, %d2, %d3, %d4)
    firrtl.int.generic "circt_mux2cell"  %sel, %d1, %d2 : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.int.generic "circt_mux4cell"  %sel2, %d1, %d2, %d3, %d4 : (!firrtl.uint<2>, !firrtl.uint, !firrtl.uint, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  }

  // CHECK-LABEL: @HasBeenReset
  firrtl.module @HasBeenReset(in %clock: !firrtl.clock, in %reset1: !firrtl.uint<1>, in %reset2: !firrtl.asyncreset, in %reset3: !firrtl.reset) {
    // CHECK-NEXT: firrtl.int.has_been_reset %clock, %reset1 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.int.has_been_reset %clock, %reset2 : !firrtl.asyncreset
    // CHECK-NEXT: firrtl.int.has_been_reset %clock, %reset3 : !firrtl.reset
    firrtl.int.generic "circt_has_been_reset"  %clock, %reset1 : (!firrtl.clock, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.int.generic "circt_has_been_reset"  %clock, %reset2 : (!firrtl.clock, !firrtl.asyncreset) -> !firrtl.uint<1>
    firrtl.int.generic "circt_has_been_reset"  %clock, %reset3 : (!firrtl.clock, !firrtl.reset) -> !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @ChiselVerif(
  firrtl.module @ChiselVerif(in %clock: !firrtl.clock,
                             in %cond: !firrtl.uint<1>,
                             in %enable: !firrtl.uint<1>) {
    // CHECK: firrtl.assert %{{.+}}, %{{.+}}, %{{.+}}, "testing" :
    // CHECK-SAME: isConcurrent = true
    firrtl.int.generic "circt_chisel_assert" <format: none = "testing"> %clock, %cond, %enable : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>) -> ()
    // CHECK: firrtl.assert %{{.+}}, %{{.+}}, %{{.+}}, "message: %d"(
    // CHECK-SAME: guards = ["MACRO_GUARD", "ASDF"]
    // CHECK-SAME: isConcurrent = true
    // CHECK-SAME: name = "label for assert with format string"
    firrtl.int.generic "circt_chisel_assert" <format: none = "message: %d", label: none = "label for assert with format string", guards: none = "MACRO_GUARD;ASDF"> %clock, %cond, %enable, %cond : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> ()
    // CHECK: firrtl.assert %{{.+}}, %{{.+}}, %{{.+}}, "ief: %d"(
    // CHECK-SAME: format = "ifElseFatal"
    // CHECK-SAME: guards = ["MACRO_GUARD", "ASDF"]
    // CHECK-SAME: isConcurrent = true
    // CHECK-SAME: name = "label for ifelsefatal assert"
    firrtl.int.generic "circt_chisel_ifelsefatal" <format: none = "ief: %d", label: none = "label for ifelsefatal assert", guards: none = "MACRO_GUARD;ASDF"> %clock, %cond, %enable, %enable : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> ()
    // CHECK: firrtl.assume %{{.+}}, %{{.+}}, %{{.+}}, "text: %d"(
    // CHECK-SAME: isConcurrent = true
    // CHECK-SAME: name = "label for assume"
    firrtl.int.generic "circt_chisel_assume" <format: none = "text: %d", label: none = "label for assume"> %clock, %cond, %enable, %enable : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> ()
    // CHECK: firrtl.cover %{{.+}}, %{{.+}}, %{{.+}}, "" :
    // CHECK-SAME: isConcurrent = true
    // CHECK-SAME: name = "label for cover"
    firrtl.int.generic "circt_chisel_cover" <label: none = "label for cover"> %clock, %cond, %enable : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>) -> ()

    // CHECK: firrtl.int.unclocked_assume %{{.+}}, %{{.+}}, "text: %d"(
    // CHECK-SAME: guards = ["MACRO_GUARD", "ASDF"]
    // CHECK-SAME: name = "label for unr"
    firrtl.int.generic "circt.unclocked_assume" <format: none = "text: %d", label: none = "label for unr", guards: none = "MACRO_GUARD;ASDF"> %cond, %enable, %enable : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> ()
  }

  // CHECK-LABEL: firrtl.module private @ProbeIntrinsicTest
  firrtl.module private @ProbeIntrinsicTest(in %clock : !firrtl.clock, in %data : !firrtl.uint<32>) {
    // CHECK-NEXT: firrtl.int.fpga_probe %clock, %data : !firrtl.uint<32>
    firrtl.int.generic "circt_fpga_probe"  %data, %clock : (!firrtl.uint<32>, !firrtl.clock) -> ()
  }

  // CHECK-LABEL: firrtl.module private @DPIIntrinsicTest
  firrtl.module private @DPIIntrinsicTest(in %clock : !firrtl.clock, in %enable : !firrtl.uint<1>, in %in1: !firrtl.uint<8>, in %in2: !firrtl.uint<8>) {
    // CHECK-NEXT: %0 = firrtl.int.dpi.call "clocked_result"(%in1, %in2) clock %clock enable %enable {inputNames = ["foo", "bar"], outputName = "baz"} : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
    %0 = firrtl.int.generic "circt_dpi_call" <isClocked: ui32 = 1, functionName: none = "clocked_result", inputNames: none = "foo;bar", outputName: none = "baz"> %clock, %enable, %in1, %in2 : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
    // CHECK-NEXT: firrtl.int.dpi.call "clocked_void"(%in1, %in2) clock %clock enable %enable : (!firrtl.uint<8>, !firrtl.uint<8>) -> ()
    firrtl.int.generic "circt_dpi_call" <isClocked: ui32 = 1, functionName: none = "clocked_void"> %clock, %enable, %in1, %in2 : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> ()
    // CHECK-NEXT:  %1 = firrtl.int.dpi.call "unclocked_result"(%in1, %in2) enable %enable : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
    %1 = firrtl.int.generic "circt_dpi_call" <isClocked: ui32 = 0, functionName: none = "unclocked_result"> %enable, %in1, %in2 : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
  }

  // CHECK-LABEL: firrtl.module private @ViewIntrinsicTest
  firrtl.module private @ViewIntrinsicTest(in %in: !firrtl.vector<uint<1>, 5>) attributes {convention = #firrtl<convention scalarized>} {
    %0 = firrtl.subindex %in[4] : !firrtl.vector<uint<1>, 5>
    %1 = firrtl.subindex %in[3] : !firrtl.vector<uint<1>, 5>
    %2 = firrtl.subindex %in[2] : !firrtl.vector<uint<1>, 5>
    %3 = firrtl.subindex %in[1] : !firrtl.vector<uint<1>, 5>
    %4 = firrtl.subindex %in[0] : !firrtl.vector<uint<1>, 5>

    // Taken from old test, these descriptions don't really apply anymore but leaving as-is for now.
    firrtl.int.generic "circt_view" <name: none = "view", info: none = "{\"class\":\"sifive.enterprise.grandcentral.AugmentedBundleType\",\"defName\":\"ViewName\",\"elements\":[{\"description\":\"the register in GCTInterface\",\"name\":\"register\",\"tpe\":{\"class\":\"sifive.enterprise.grandcentral.AugmentedBundleType\",\"defName\":\"Register\",\"elements\":[{\"name\":\"_2\",\"tpe\":{\"class\":\"sifive.enterprise.grandcentral.AugmentedVectorType\",\"elements\":[{\"class\":\"sifive.enterprise.grandcentral.AugmentedGroundType\"},{\"class\":\"sifive.enterprise.grandcentral.AugmentedGroundType\"}]}},{\"name\":\"_0_inst\",\"tpe\":{\"class\":\"sifive.enterprise.grandcentral.AugmentedBundleType\",\"defName\":\"_0_def\",\"elements\":[{\"name\":\"_1\",\"tpe\":{\"class\":\"sifive.enterprise.grandcentral.AugmentedGroundType\"}},{\"name\":\"_0\",\"tpe\":{\"class\":\"sifive.enterprise.grandcentral.AugmentedGroundType\"}}]}}]}},{\"description\":\"the port 'a' in GCTInterface\",\"name\":\"port\",\"tpe\":{\"class\":\"sifive.enterprise.grandcentral.AugmentedGroundType\"}}]}"> %4, %3, %2, %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> ()

    // CHECK: firrtl.view "view", <{
    // CHECK-SAME:   class = "sifive.enterprise.grandcentral.AugmentedBundleType",
    // CHECK-SAME:   defName = "ViewName",
    // CHECK-SAME:   elements = [
    // CHECK-SAME:      {
    // CHECK-SAME:        class = "sifive.enterprise.grandcentral.AugmentedBundleType",
    // CHECK-SAME:        defName = "Register",
    // CHECK-SAME:        description = "the register in GCTInterface",
    // CHECK-SAME:        elements = [
    // CHECK-SAME:          {
    // CHECK-SAME:            class = "sifive.enterprise.grandcentral.AugmentedVectorType",
    // CHECK-SAME:            elements = [
    // CHECK-SAME:              {
    // CHECK-SAME:                class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    // CHECK-SAME:                name = "_2"
    // CHECK-SAME:              }, {
    // CHECK-SAME:                class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    // CHECK-SAME:                name = "_2
    // CHECK-SAME:              }
    // CHECK-SAME:            ],
    // CHECK-SAME:            name = "_2"
    // CHECK-SAME:          }, {
    // CHECK-SAME:            class = "sifive.enterprise.grandcentral.AugmentedBundleType",
    // CHECK-SAME:            defName = "_0_def",
    // CHECK-SAME:            elements = [
    // CHECK-SAME:              {
    // CHECK-SAME:                class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    // CHECK-SAME:                name = "_1"
    // CHECK-SAME:              }, {
    // CHECK-SAME:                class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    // CHECK-SAME:                name = "_0"
    // CHECK-SAME:              }
    // CHECK-SAME:            ],
    // CHECK-SAME:            name = "_0_inst"
    // CHECK-SAME:          }
    // CHECK-SAME:        ],
    // CHECK-SAME:        name = "register"
    // CHECK-SAME:     }, {
    // CHECK-SAME:       class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    // CHECK-SAME:       description = "the port 'a' in GCTInterface",
    // CHECK-SAME:       name = "port"
    // CHECK-SAME:     }
    // CHECK-SAME:   ],
    // This is copied in, for better or for worse.
    // CHECK-SAME:   name = "view"
    // CHECK-SAME: }>,
    // Check operands and types.
    // CHECK-SAME: %4, %3, %2, %1, %0 : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>

    firrtl.int.generic "circt_view" <name: none = "view2", yaml: none = "views.yml", info: none = "{\"class\":\"sifive.enterprise.grandcentral.AugmentedBundleType\",\"defName\":\"ViewName\",\"elements\":[]}"> : () -> ()

    // CHECK: firrtl.view "view2", yaml "views.yml", <{
  }
}
