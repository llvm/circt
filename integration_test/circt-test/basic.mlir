// See other `basic-*.mlir` files for run lines.
// RUN: true

// CHECK: 1 tests FAILED, 9 passed, 1 ignored, 3 unsupported

hw.module @FullAdder(in %a: i1, in %b: i1, in %ci: i1, out s: i1, out co: i1) {
  %0 = comb.xor %a, %b : i1
  %1 = comb.xor %0, %ci : i1
  %2 = comb.and %a, %b : i1
  %3 = comb.and %0, %ci : i1
  %4 = comb.or %2, %3 : i1
  hw.output %1, %4 : i1, i1
}

hw.module @CustomAdderWithCarry(in %a: i4, in %b: i4, in %ci: i1, out z: i4, out co: i1) {
  %a0 = comb.extract %a from 0 : (i4) -> i1
  %a1 = comb.extract %a from 1 : (i4) -> i1
  %a2 = comb.extract %a from 2 : (i4) -> i1
  %a3 = comb.extract %a from 3 : (i4) -> i1
  %b0 = comb.extract %b from 0 : (i4) -> i1
  %b1 = comb.extract %b from 1 : (i4) -> i1
  %b2 = comb.extract %b from 2 : (i4) -> i1
  %b3 = comb.extract %b from 3 : (i4) -> i1
  %adder0.s, %adder0.co = hw.instance "adder0" @FullAdder(a: %a0: i1, b: %b0: i1, ci: %ci: i1) -> (s: i1, co: i1)
  %adder1.s, %adder1.co = hw.instance "adder1" @FullAdder(a: %a1: i1, b: %b1: i1, ci: %adder0.co: i1) -> (s: i1, co: i1)
  %adder2.s, %adder2.co = hw.instance "adder2" @FullAdder(a: %a2: i1, b: %b2: i1, ci: %adder1.co: i1) -> (s: i1, co: i1)
  %adder3.s, %adder3.co = hw.instance "adder3" @FullAdder(a: %a3: i1, b: %b3: i1, ci: %adder2.co: i1) -> (s: i1, co: i1)
  %z = comb.concat %adder3.s, %adder2.s, %adder1.s, %adder0.s : i1, i1, i1, i1
  hw.output %z, %adder3.co : i4, i1
}

hw.module @CustomAdder(in %a: i4, in %b: i4, out z: i4) {
  %false = hw.constant false
  %z, %co = hw.instance "adder" @CustomAdderWithCarry(a: %a: i4, b: %b: i4, ci: %false: i1) -> (z: i4, co: i1)
  hw.output %z : i4
}

verif.formal @ZeroLhs {} {
  %c0_i4 = hw.constant 0 : i4
  %x = verif.symbolic_value : i4
  %z = hw.instance "dut" @CustomAdder(a: %c0_i4: i4, b: %x: i4) -> (z: i4)
  %eq = comb.icmp eq %z, %x : i4
  verif.assert %eq : i1
}

verif.formal @ZeroRhs {} {
  %c0_i4 = hw.constant 0 : i4
  %x = verif.symbolic_value : i4
  %z = hw.instance "dut" @CustomAdder(a: %x: i4, b: %c0_i4: i4) -> (z: i4)
  %eq = comb.icmp eq %z, %x : i4
  verif.assert %eq : i1
}

verif.formal @CustomAdderWorks {} {
  %a = verif.symbolic_value : i4
  %b = verif.symbolic_value : i4

  // Check against reference adder.
  %z0 = hw.instance "dut" @CustomAdder(a: %a: i4, b: %b: i4) -> (z: i4)
  %z1 = comb.add %a, %b : i4
  %eq = comb.icmp eq %z0, %z1 : i4
  verif.assert %eq : i1

  // Show this can compute 5 somehow.
  %c5_i4 = hw.constant 5 : i4
  %0 = comb.icmp eq %z0, %c5_i4 : i4
  %1 = comb.icmp ne %a, %b : i4
  %2 = comb.and %0, %1 : i1
  // verif.cover %2 : i1  // not supported in circt-bmc currently
}

hw.module @ALU(in %a: i4, in %b: i4, in %sub: i1, out z: i4) {
  %sub_mask = comb.replicate %sub : (i1) -> i4
  %b2 = comb.xor %b, %sub_mask : i4
  %z, %co = hw.instance "adder" @CustomAdderWithCarry(a: %a: i4, b: %b2: i4, ci: %sub: i1) -> (z: i4, co: i1)
  hw.output %z : i4
}

verif.formal @ALUCanAdd {} {
  %a = verif.symbolic_value : i4
  %b = verif.symbolic_value : i4
  %false = hw.constant false
  %z0 = hw.instance "dut" @ALU(a: %a: i4, b: %b: i4, sub: %false: i1) -> (z: i4)
  %z1 = comb.add %a, %b : i4
  %eq = comb.icmp eq %z0, %z1 : i4
  verif.assert %eq : i1
}

verif.formal @ALUCanSub {mode = "cover,induction"} {
  %a = verif.symbolic_value : i4
  %b = verif.symbolic_value : i4
  %true = hw.constant true
  %z0 = hw.instance "dut" @ALU(a: %a: i4, b: %b: i4, sub: %true: i1) -> (z: i4)
  %z1 = comb.sub %a, %b : i4
  %eq = comb.icmp eq %z0, %z1 : i4
  verif.assert %eq : i1
}

verif.formal @ALUWorks {mode = "cover,bmc", depth = 5} {
  %a = verif.symbolic_value : i4
  %b = verif.symbolic_value : i4
  %sub = verif.symbolic_value : i1

  // Custom ALU implementation.
  %z0 = hw.instance "dut" @ALU(a: %a: i4, b: %b: i4, sub: %sub: i1) -> (z: i4)

  // Reference add/sub function.
  %ref_add = comb.add %a, %b : i4
  %ref_sub = comb.sub %a, %b : i4
  %z1 = comb.mux %sub, %ref_sub, %ref_add : i4

  // Check the two match.
  %eq = comb.icmp eq %z0, %z1 : i4
  verif.assert %eq : i1
}

verif.formal @ALUIgnoreFailure {ignore = true} {
  %a = verif.symbolic_value : i4
  %b = verif.symbolic_value : i4
  %sub = verif.symbolic_value : i1

  // Custom ALU implementation.
  %z0 = hw.instance "dut" @ALU(a: %a: i4, b: %b: i4, sub: %sub: i1) -> (z: i4)

  // Reference add/sub function.
  %ref_add = comb.add %a, %b : i4
  %ref_sub = comb.sub %a, %b : i4
  %z1 = comb.mux %sub, %ref_sub, %ref_add : i4

  // Check the two don't match (failure will be ignored).
  %ne = comb.icmp ne %z0, %z1 : i4
  verif.assert %ne : i1
}

verif.formal @ALUFailure {depth = 3} {
  %a = verif.symbolic_value : i4
  %b = verif.symbolic_value : i4
  %sub = verif.symbolic_value : i1

  // Custom ALU implementation.
  %z0 = hw.instance "dut" @ALU(a: %a: i4, b: %b: i4, sub: %sub: i1) -> (z: i4)

  // Reference add/sub function.
  %ref_add = comb.add %a, %b : i4
  %ref_sub = comb.sub %a, %b : i4
  %z1 = comb.mux %sub, %ref_sub, %ref_add : i4

  // Check the two don't match (should fail).
  %ne = comb.icmp ne %z0, %z1 : i4
  verif.assert %ne : i1
}

verif.formal @RunnerRequireEither {require_runners = ["sby", "circt-bmc"]} {
  %0 = hw.constant true
  verif.cover %0 : i1
}

verif.formal @RunnerRequireSby {require_runners = ["sby"]} {
  %0 = hw.constant true
  verif.cover %0 : i1
}

verif.formal @RunnerRequireCirctBmc {require_runners = ["circt-bmc"]} {
  %0 = hw.constant true
  verif.cover %0 : i1
}

verif.formal @RunnerExcludeEither {exclude_runners = ["sby", "circt-bmc"]} {
  %0 = hw.constant true
  verif.cover %0 : i1
}

verif.formal @RunnerExcludeSby {exclude_runners = ["sby"]} {
  %0 = hw.constant true
  verif.cover %0 : i1
}

verif.formal @RunnerExcludeCirctBmc {exclude_runners = ["circt-bmc"]} {
  %0 = hw.constant true
  verif.cover %0 : i1
}
