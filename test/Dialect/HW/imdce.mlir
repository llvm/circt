// RUN: circt-opt -hw-imdce=print-liveness --split-input-file --allow-unregistered-dialect %s | FileCheck %s --check-prefixes=LIVENESS

module {

  // ===== Dead uninstantiated module =====
  // - Private module with no instantiations

  // LIVENESS-LABEL: hw.module private @dead_uninstantiated
  // LIVENESS-SAME: "op-liveness" = "DEAD"
  // LIVENESS-SAME: "val-liveness" = []
  hw.module private @dead_uninstantiated() {
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = []
    hw.output
  }

  // ===== Dead instantiated module =====
  // - @dead_module is instantiated but its output is unused
  // - @dead_module_dead_user is public, so it's executable

  // LIVENESS-LABEL: hw.module private @dead_module
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["DEAD", "DEAD"]
  hw.module private @dead_module(in %source1 : i1, in %source2 : i1, out dest : i1) {
    // LIVENESS: comb.and
    // LIVENESS-SAME: "op-liveness" = "DEAD"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %and = comb.and %source1, %source2 : i1
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    hw.output %and : i1
  }

  // LIVENESS-LABEL: hw.module public @dead_module_dead_user
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = []
  hw.module public @dead_module_dead_user(out const : i1) {
    // LIVENESS: hw.constant false
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    %s1 = hw.constant 0 : i1
    // LIVENESS: hw.constant true
    // LIVENESS-SAME: "op-liveness" = "DEAD"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %s2 = hw.constant 1 : i1
    // LIVENESS: hw.instance "dead_module_instance"
    // LIVENESS-SAME: "op-liveness" = "DEAD"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %dest = hw.instance "dead_module_instance" @dead_module(source1: %s1 : i1, source2 : %s2 : i1) -> (dest : i1)
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %s1 : i1
  }

  // ===== Dead port on alive module =====
  // - @public_with_dead_port is public, so its output is live
  // - This propagates liveness to %source2, but %source1 is unused (dead)
  // - @dead_port_alive_module is private but instantiated with live output

  // LIVENESS-LABEL: hw.module private @dead_port_alive_module
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["DEAD", "LIVE"]
  hw.module private @dead_port_alive_module(in %source1 : i1, in %source2 : i1, out dest : i1) {
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %source2 : i1
  }

  // LIVENESS-LABEL: hw.module public @public_with_dead_port
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["DEAD", "LIVE"]
  hw.module public @public_with_dead_port(in %source1 : i1, in %source2 : i1, out dest : i1) {
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %source2 : i1
  }

  // LIVENESS-LABEL: hw.module public @dead_port_alive_module_user
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = []
  hw.module public @dead_port_alive_module_user(out dest1 : i1, out dest2 : i1) {
    // LIVENESS: hw.constant false
    // LIVENESS-SAME: "op-liveness" = "DEAD"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %s1 = hw.constant 0 : i1
    // LIVENESS: hw.constant true
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    %s2 = hw.constant 1 : i1
    // LIVENESS: hw.instance "dead_port_alive_module_instance"
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    %dest1 = hw.instance "dead_port_alive_module_instance" @dead_port_alive_module(source1: %s1 : i1, source2 : %s2 : i1) -> (dest : i1)
    // LIVENESS: hw.instance "public_with_dead_port_instance"
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    %dest2 = hw.instance "public_with_dead_port_instance" @public_with_dead_port(source1: %s1 : i1, source2 : %s2 : i1) -> (dest : i1)
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE", "LIVE"]
    hw.output %dest1, %dest2 : i1, i1
  }

  // ===== Side effects keep module alive =====
  // - @Child1 has no side effects and output is unused -> dead instance
  // - @Child2 has seq.firreg (side effect) -> block executable, but instance not marked alive
  //   (instance is only alive if: has inner sym, targets external, or result is used)

  // LIVENESS-LABEL: hw.module private @Child1
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["DEAD"]
  hw.module private @Child1(in %input : i1, out output : i1) {
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    hw.output %input : i1
  }

  // LIVENESS-LABEL: hw.module private @Child2
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE", "LIVE"]
  hw.module private @Child2(in %input : i1, in %clock : !seq.clock, out output: i1) {
    // LIVENESS: seq.firreg
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    %r = seq.firreg %input clock %clock {firrtl.random_init_start = 0 : ui64} : i1
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %r : i1
  }

  // LIVENESS-LABEL: hw.module public @Top
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["DEAD", "DEAD"]
  hw.module public @Top(in %clock: !seq.clock, in %input : i1) {
    // LIVENESS: hw.instance "child1_instance"
    // LIVENESS-SAME: "op-liveness" = "DEAD"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %child1 = hw.instance "child1_instance" @Child1(input : %input : i1) -> (output : i1)
    // LIVENESS: hw.instance "child2_instance"
    // LIVENESS-SAME: "op-liveness" = "DEAD"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %child2 = hw.instance "child2_instance" @Child2(input : %input : i1, clock : %clock : !seq.clock) -> (output : i1)
    hw.output
  }

  // ===== Unused output port =====
  // - @SingleDriver has output 'c' that is never used by any instance
  // - Output 'b' is used, so it and its dependencies are live

  // LIVENESS-LABEL: hw.module private @SingleDriver
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE"]
  hw.module private @SingleDriver(in %a : i1, out b : i1, out c : i1) {
    // LIVENESS: hw.constant false
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    %false = hw.constant false
    // LIVENESS: comb.and
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    %0 = comb.and %a, %false : i1
    // LIVENESS: hw.constant true
    // LIVENESS-SAME: "op-liveness" = "DEAD"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %true = hw.constant true
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE", "DEAD"]
    hw.output %0, %true : i1, i1
  }

  // LIVENESS-LABEL: hw.module public @UnusedOutput
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE"]
  hw.module public @UnusedOutput(in %a : i1, out b : i1) {
    // LIVENESS: hw.instance "SingleDriverInstance"
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE", "DEAD"]
    %sd_b, %sd_c = hw.instance "SingleDriverInstance" @SingleDriver (a : %a : i1) -> (b : i1, c : i1)
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %sd_b : i1
  }

  // ===== Empty modules and symbols =====
  // - @empty has no ports and no body
  // - @Sub has dead input (not used internally)
  // - Instance "sub" has sym @Foo, which keeps it alive

  hw.module private @empty() attributes {annotations = [{class = "foo"}]} {}

  // LIVENESS-LABEL: hw.module private @Sub
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["DEAD"]
  hw.module private @Sub(in %a: i1)  {}

  // LIVENESS-LABEL: hw.module public @DeleteEmptyModule
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = []
  hw.module public @DeleteEmptyModule() {
    // LIVENESS: hw.constant true
    // LIVENESS-SAME: "op-liveness" = "DEAD"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %a = hw.constant true
    hw.instance "sub" sym @Foo @Sub(a: %a : i1) -> ()
    hw.instance "sub2" @Sub(a: %a: i1) -> ()
    hw.instance "empty" @empty() -> ()
    hw.output
  }

  // ===== Multiple dead ports indexing =====
  // - Verifies that port removal indexing works when multiple ports are dead

  // LIVENESS-LABEL: hw.module private @multiple_remove_ports
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["DEAD", "DEAD", "LIVE"]
  hw.module private @multiple_remove_ports(in %source1 : i1, in %source2 : i1, in %source3 : i1, out result : i1) {
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %source3 : i1
  }

  // LIVENESS-LABEL: hw.module public @mrp_user
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE"]
  hw.module public @mrp_user(in %in : i1, out result : i1) {
    // Result used -> instance alive
    // LIVENESS: hw.instance "mrp_instance"
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    %out = hw.instance "mrp_instance" @multiple_remove_ports(source1: %in: i1, source2: %in: i1, source3: %in: i1) -> (result: i1)
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %out : i1
  }

  // ===== Dead ops in body =====
  // - Pure ops with unused results should be dead
  // - Ops with side effects (seq.firreg) have live results (and mark operands live)

  // LIVENESS-LABEL: hw.module public @DeadOpsInBody
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE", "LIVE"]
  hw.module public @DeadOpsInBody(in %a: i1, in %clk: !seq.clock, out result: i1) {
    // LIVENESS: seq.firreg
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    %dead_reg = seq.firreg %a clock %clk : i1
    // LIVENESS: comb.and
    // LIVENESS-SAME: "op-liveness" = "DEAD"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %dead_and = comb.and %a, %a : i1
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %a : i1
  }

  // ===== Nested hierarchy liveness propagation =====
  // - Liveness propagates through: NestedTop -> NestedMiddle -> NestedBottom
  // - Only the 'y' output path is live; 'dead' output path is dead

  // LIVENESS-LABEL: hw.module private @NestedBottom
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE"]
  hw.module private @NestedBottom(in %x: i1, out y: i1, out dead: i1) {
    // LIVENESS: hw.constant false
    // LIVENESS-SAME: "op-liveness" = "DEAD"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %c = hw.constant false
    // LIVENESS: comb.and
    // LIVENESS-SAME: "op-liveness" = "DEAD"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %unused_and = comb.and %c, %x : i1
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE", "DEAD"]
    hw.output %x, %unused_and : i1, i1
  }

  // LIVENESS-LABEL: hw.module private @NestedMiddle
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE"]
  hw.module private @NestedMiddle(in %x: i1, out y: i1) {
    // LIVENESS: hw.instance "bot"
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE", "DEAD"]
    %y, %dead = hw.instance "bot" @NestedBottom(x: %x: i1) -> (y: i1, dead: i1)
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %y : i1
  }

  // LIVENESS-LABEL: hw.module public @NestedTop
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE"]
  hw.module public @NestedTop(in %x: i1, out y: i1) {
    // LIVENESS: hw.instance "mid"
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    %y = hw.instance "mid" @NestedMiddle(x: %x: i1) -> (y: i1)
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %y : i1
  }

  // ===== Side effect operands are live =====
  // - seq.firreg uses %reset and %a as operands
  // - These operands should be marked live because they feed a side-effect op

  // LIVENESS-LABEL: hw.module public @SideEffectPreserved
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE", "LIVE"]
  hw.module public @SideEffectPreserved(in %a: i1, in %clk: !seq.clock, out result: i1) {
    // LIVENESS: hw.constant true
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    %reset = hw.constant true
    // LIVENESS: seq.firreg
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    %live_reg = seq.firreg %a clock %clk reset sync %reset, %a : i1
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %a : i1
  }

  // ===== Shared module with different users =====
  // - @SharedModule is used by two modules
  // - SharedUser1 uses output 'x', SharedUser2 uses output 'y'
  // - Both inputs/outputs must be kept live because of cross-user dependencies

  // LIVENESS-LABEL: hw.module private @SharedModule
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE", "LIVE"]
  hw.module private @SharedModule(in %a: i1, in %b: i1, out x: i1, out y: i1) {
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE", "LIVE"]
    hw.output %a, %b : i1, i1
  }

  // LIVENESS-LABEL: hw.module public @SharedUser1
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE"]
  hw.module public @SharedUser1(in %in: i1, out out: i1) {
    // LIVENESS: hw.instance "s1"
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE", "DEAD"]
    %x, %y = hw.instance "s1" @SharedModule(a: %in: i1, b: %in: i1) -> (x: i1, y: i1)
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %x : i1
  }

  // LIVENESS-LABEL: hw.module public @SharedUser2
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE"]
  hw.module public @SharedUser2(in %in: i1, out out: i1) {
    // LIVENESS: hw.instance "s2"
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["DEAD", "LIVE"]
    %x, %y = hw.instance "s2" @SharedModule(a: %in: i1, b: %in: i1) -> (x: i1, y: i1)
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %y : i1
  }

  // ===== Mixed dead/live ops =====
  // - Multiple pure ops, only one feeds the output

  // LIVENESS-LABEL: hw.module public @DeadOpsVariety
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE", "LIVE", "DEAD"]
  hw.module public @DeadOpsVariety(in %a: i1, in %b: i1, in %clk: !seq.clock, out result: i1) {
    // LIVENESS: comb.or
    // LIVENESS-SAME: "op-liveness" = "DEAD"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %dead1 = comb.or %a, %b : i1
    // LIVENESS: comb.xor
    // LIVENESS-SAME: "op-liveness" = "DEAD"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %dead2 = comb.xor %a, %b : i1
    // LIVENESS: comb.and
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    %live = comb.and %a, %b : i1
    // LIVENESS: comb.add
    // LIVENESS-SAME: "op-liveness" = "DEAD"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %dead3 = comb.add %a, %b : i1
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %live : i1
  }

  // ===== Dead constant chain =====
  // - Constant and op only feed dead operations

  // LIVENESS-LABEL: hw.module public @DeadConstantChain
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE"]
  hw.module public @DeadConstantChain(in %a: i1, out result: i1) {
    // LIVENESS: hw.constant false
    // LIVENESS-SAME: "op-liveness" = "DEAD"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %c0 = hw.constant false
    // LIVENESS: comb.and
    // LIVENESS-SAME: "op-liveness" = "DEAD"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %dead = comb.and %c0, %a : i1
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %a : i1
  }

  // ===== Alternating live/dead ports =====
  // - Tests that port removal indexing handles non-contiguous patterns

  // LIVENESS-LABEL: hw.module private @AlternatingPorts
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE", "DEAD", "LIVE", "DEAD"]
  hw.module private @AlternatingPorts(
    in %live1: i1, in %dead1: i1, in %live2: i1, in %dead2: i1,
    out out1: i1, out dead_out1: i1, out out2: i1, out dead_out2: i1
  ) {
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE", "DEAD", "LIVE", "DEAD"]
    hw.output %live1, %dead1, %live2, %dead2 : i1, i1, i1, i1
  }

  // LIVENESS-LABEL: hw.module public @AlternatingUser
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE"]
  hw.module public @AlternatingUser(in %x: i1, out r1: i1, out r2: i1) {
    // LIVENESS: hw.instance "alt"
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE", "DEAD", "LIVE", "DEAD"]
    %o1, %d1, %o2, %d2 = hw.instance "alt" @AlternatingPorts(
      live1: %x: i1, dead1: %x: i1, live2: %x: i1, dead2: %x: i1
    ) -> (out1: i1, dead_out1: i1, out2: i1, dead_out2: i1)
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE", "LIVE"]
    hw.output %o1, %o2 : i1, i1
  }

  // ===== External modules =====
  // - External modules have unknown implementations
  // - Instances targeting externals are conservatively kept alive

  hw.module.extern private @ExternalMod(in %a: i1, out b: i1)

  // LIVENESS-LABEL: hw.module public @UseExternal
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE"]
  hw.module public @UseExternal(in %x: i1, out result: i1) {
    // LIVENESS: hw.instance "ext"
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    %b = hw.instance "ext" @ExternalMod(a: %x: i1) -> (b: i1)
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %b : i1
  }

  hw.module.extern private @UnusedExternal(in %a: i1, out b: i1)

  // LIVENESS-LABEL: hw.module public @DeadExternal
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE"]
  hw.module public @DeadExternal(in %x: i1) {
    // LIVENESS: hw.instance "dead_ext"
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %dead = hw.instance "dead_ext" @UnusedExternal(a: %x: i1) -> (b: i1)
    hw.output
  }

  // ===== hw.instance_choice as HWInstanceLike =====
  // - hw.instance_choice is HWInstanceLike
  // - The pass must propagate all through options
  // - Both choices are live

  // LIVENESS-LABEL: hw.module private @ChoiceA
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE"]
  hw.module private @ChoiceA(in %x: i1, out y: i1) {
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %x : i1
  }

  // LIVENESS-LABEL: hw.module private @ChoiceB
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE"]
  hw.module private @ChoiceB(in %x: i1, out y: i1) {
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %x : i1
  }

  // LIVENESS-LABEL: hw.module public @InstanceChoiceUser
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE"]
  hw.module public @InstanceChoiceUser(in %a: i1, out result: i1) {
    // LIVENESS: hw.instance_choice "choice_inst"
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    %y = hw.instance_choice "choice_inst" option "opt" @ChoiceA or @ChoiceB if "B" (x: %a: i1) -> (y: i1)
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %y : i1
  }

  // ===== hw.instance_choice with extern choice: conservative liveness =====
  // - When any choice targets a non-HWModuleOp HWModuleLike (hw.module.extern),
  //   markInstanceLike takes the conservative path: the instance and all its
  //   operands are marked alive regardless of output use.
  // - @ChoiceNative's body becomes executable but its input stays DEAD because
  //   no live instance result is ever traced back through it.

  hw.module.extern private @ChoiceExtern(in %x: i1, out y: i1)

  // LIVENESS-LABEL: hw.module private @ChoiceNative
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["DEAD"]
  hw.module private @ChoiceNative(in %x: i1, out y: i1) {
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    hw.output %x : i1
  }

  // LIVENESS-LABEL: hw.module public @MixedChoiceUser
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE"]
  hw.module public @MixedChoiceUser(in %a: i1) {
    // LIVENESS: hw.instance_choice "mixed_inst"
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %y = hw.instance_choice "mixed_inst" option "opt" @ChoiceNative or @ChoiceExtern if "ext" (x: %a: i1) -> (y: i1)
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = []
    hw.output
  }

  // ===== Side-effect ops reached via HWInstanceLike dispatch =====
  // - markBlockExecutable dispatches to markInstanceLike for HWInstanceLike ops,
  //   which in turn calls markBlockExecutable on each referenced module body.
  // - A seq.firreg inside @SideEffectChoice forces its operands (%x, %clk)
  //   and result alive via markUnknownSideEffectOp.
  // - However, "se_inst" itself is never added to liveElements (its output is
  //   unused and both choices are HWModuleOp, so the conservative extern path
  //   is not taken). visitInstanceLike therefore never runs for "se_inst", so
  //   the internal liveness of @SideEffectChoice does NOT propagate back to
  //   the instance operands %a / %c in @SideEffectChoiceUser.

  // LIVENESS-LABEL: hw.module private @SideEffectChoice
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE", "LIVE"]
  hw.module private @SideEffectChoice(in %x: i1, in %clk: !seq.clock, out y: i1) {
    // LIVENESS: seq.firreg
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    %r = seq.firreg %x clock %clk {firrtl.random_init_start = 0 : ui64} : i1
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %r : i1
  }

  // LIVENESS-LABEL: hw.module private @PureChoice
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["DEAD", "DEAD"]
  hw.module private @PureChoice(in %x: i1, in %clk: !seq.clock, out y: i1) {
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    hw.output %x : i1
  }

  // LIVENESS-LABEL: hw.module public @SideEffectChoiceUser
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["DEAD", "DEAD"]
  hw.module public @SideEffectChoiceUser(in %a: i1, in %c: !seq.clock) {
    // LIVENESS: hw.instance_choice "se_inst"
    // LIVENESS-SAME: "op-liveness" = "DEAD"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %y = hw.instance_choice "se_inst" option "opt" @SideEffectChoice or @PureChoice if "P" (x: %a: i1, clk: %c: !seq.clock) -> (y: i1)
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = []
    hw.output
  }

  // ===== Dead instance does not propagate input liveness =====
  // - @Gated is instantiated by both @GatedUser (live instance, output used)
  //   and @GatedDead (dead instance, output discarded).
  // - @Gated's %x is alive because visitValue propagates from the live result
  //   of "g_live" back through @Gated's body terminator.
  // - When %x is then visited as a block arg, the pass checks
  //   liveElements.contains(instance) before back-propagating to the operand.
  //   "g_dead" is not in liveElements, so %b in @GatedDead stays DEAD.

  // LIVENESS-LABEL: hw.module private @Gated
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE"]
  hw.module private @Gated(in %x: i1, out y: i1) {
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %x : i1
  }

  // LIVENESS-LABEL: hw.module public @GatedUser
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE"]
  hw.module public @GatedUser(in %a: i1, out result: i1) {
    // LIVENESS: hw.instance "g_live"
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    %y = hw.instance "g_live" @Gated(x: %a: i1) -> (y: i1)
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %y : i1
  }

  // LIVENESS-LABEL: hw.module public @GatedDead
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["DEAD"]
  hw.module public @GatedDead(in %b: i1) {
    // LIVENESS: hw.instance "g_dead"
    // LIVENESS-SAME: "op-liveness" = "DEAD"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %y = hw.instance "g_dead" @Gated(x: %b: i1) -> (y: i1)
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = []
    hw.output
  }

  // ===== HWModuleLike (extern): all instances conservatively alive =====
  // - hw.module.extern implements HWModuleLike but not HWModuleOp.
  // - markInstanceLike conservatively marks every instance of a non-HWModuleOp
  //   target alive and marks all its operands alive, regardless of output use.
  // - "ext_live" has its result consumed; "ext_dead" does not — both are LIVE.

  hw.module.extern private @MultiExt(in %a: i1, out b: i1)

  // LIVENESS-LABEL: hw.module public @MultiExtUser
  // LIVENESS-SAME: "op-liveness" = "LIVE"
  // LIVENESS-SAME: "val-liveness" = ["LIVE", "LIVE"]
  hw.module public @MultiExtUser(in %x: i1, in %y: i1, out result: i1) {
    // LIVENESS: hw.instance "ext_live"
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    %b1 = hw.instance "ext_live" @MultiExt(a: %x: i1) -> (b: i1)
    // LIVENESS: hw.instance "ext_dead"
    // LIVENESS-SAME: "op-liveness" = "LIVE"
    // LIVENESS-SAME: "val-liveness" = ["DEAD"]
    %b2 = hw.instance "ext_dead" @MultiExt(a: %y: i1) -> (b: i1)
    // LIVENESS: hw.output
    // LIVENESS-SAME: "val-liveness" = ["LIVE"]
    hw.output %b1 : i1
  }
}
