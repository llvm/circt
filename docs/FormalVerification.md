# Formal Verification Tooling

Formally verifying hardware designs is a crucial step during the development
process. Various techniques exist, such as logical equivalence checking, model
checking, symbolic execution, etc. The preferred technique depends on the level
of abstraction, the kind of properties to be verified, runtime limitations, etc.
As a hardware compiler collection, CIRCT provides infrastructure to implement
formal verification tooling and already comes with a few tools for common
use-cases. This document provides an introduction to those tools and gives and
overview over the underlying infrastructure for compiler engineers who want to
use CIRCT to implement their custom verification tool.

[TOC]

## Logical Equivalence Checking

The `circt-lec` tool takes one or two MLIR input files with operations of the
HW, Comb, and Seq dialects and the names of two modules to check for
equivalence. To get to know the exact CLI command type `circt-lec --help`.

For example, the below IR contains two modules `@mulByTwo` and `@add`. Calling
`circt-lec input.mlir -c1=mulByTwo -c2=add` will output `c1 == c2` since they
are semantically equivalent.

```mlir
// input.mlir
hw.module @mulByTwo(in %in: i32, out out: i32) {
  dbg.variable "in", %in : i32
  %two = hw.constant 2 : i32
  dbg.variable "two", %two : i32
  %res = comb.mul %in, %two : i32
  hw.output %res : i32
}
hw.module @add(in %in: i32, out out: i32) {
  dbg.variable "in", %in : i32
  %res = comb.add %in, %in : i32
  hw.output %res : i32
}
```

In the above example, the MLIR file is compiled to LLVM IR which is then passed
to the LLVM JIT compiler to directly output the LEC result. Alternatively, there
are command line flags to print the LEC problem in SMT-LIB, LLVM IR, or an
object file that can be linked against the Z3 SMT solver to produce a standalone
binary.

## Bounded Model Checking

The `circt-bmc` tool takes one MLIR input file with operations of the HW, Comb,
and Seq dialects, along with operations from the Verif dialect to annotate the
design with properties. The name of the top module is also taken as an argument,
along with a bound indicating how many timesteps the design should be model
checked over. Similarly to `circt-lec`, the problem can be emitted as LLVM IR
for later compilation or can be passed to the LLVM JIT compiler and executed
immediately (in which case the location of a Z3 library is also required). To
get to know the exact CLI command type `circt-bmc --help`.

An example of a file that can be provided to `circt-bmc` can be seen below:

```mlir
hw.module @MyModule(in %clk: !seq.clock, in %i0: i1) {
  %c-1_i1 = hw.constant -1 : i1
  %reg = seq.compreg %i0, %clk : i1
  %not_reg = comb.xor bin %reg, %c-1_i1 : i1
  %not_not_reg = comb.xor bin %not_reg, %c-1_i1 : i1
  // Condition
  %eq = comb.icmp bin eq %not_not_reg, %reg : i1
  verif.assert %eq : i1
}
```

This design negates a register's output twice and checks that the register's
output is always equivalent to the double-negated value - this is specified with
a `verif.assert` operation. Calling
`circt-bmc -b 10 --module MyModule --shared-libs=<z3 library>` on this IR will
emit `Bound reached with no violations!`, as the property holds over the 10
cycles checked.

## Infrastructure Overview

This section provides and overview over the relevant dialects and passes for
building formal verification tools based on CIRCT. The
[below diagram](/includes/img/smt_based_formal_verification_infra.svg) provides
a visual overview.

<p align="center"><img src="https://circt.llvm.org/includes/img/smt_based_formal_verification_infra.svg"/></p>

The overall flow will insert an explicit operation to specify the verification
problem (e.g., `verif.lec`, `verif.bmc`). This operation could then be lowered
to an encoding in SMT, an interactive theorem prover, a BDD, or potentially
being exported to existing tools (currently only SMT is supported). Each of
those might have their own different backend paths as well. E.g., an encoding in
SMT can be exported to SMT-LIB or lowered to LLVM IR that calls the Z3 solver.

In the following sections we are going to explain this lowering hierarchy in
more detail by looking at an example of SMT based LEC.

### Building a custom LEC tool

Let's suppose we want to implement our own custom LEC tool for our novel HDL
which lowers to a CIRCT/MLIR based IR. There are two ways to implement this:
1. Lower this IR to the CIRCT core representation (HW, Comb, Seq) and just use
the same pipeline as `circt-lec`. This has the advantage that we don't need to
think about SMT encodings and we might already compile to those dialects anyway
for Verilog emission.
2. Implement a pass that sets up the LEC problem (lowering to the `verif.lec`
operation), and another pass that encodes the IRs operations and types in SMT
(i.e., a lowering pass to the SMT dialect). This has the advantage that
higher-level information from the IR could be taken advantage of to provide a
more efficient SMT encoding.

Consider the code example from the
[Logical Equivalence Checking section](#Logical-Equivalence-Checking) above. In
the first phase, the input IR is transformed to explicitly represent the LEC
statement using the `verif.lec` operation leading to the following IR.

```mlir
// return values indicate:
// * equivalent?
// * counter-example available?
// * counter-example value for input (undefined if not availbable)
%0:3 = verif.lec first {
^bb0(%in: i32):
  dbg.variable "in", %in : i32
  %c2_i32 = hw.constant 2 : i32
  dbg.variable "two", %c2_i32 : i32
  %0 = comb.mul %in, %c2_i32 : i32
  verif.yield %0 : i32
} second { 
^bb0(%in: i32):
  dbg.variable "in", %in : i32
  %0 = comb.add %in, %in : i32
  verif.yield %0 : i32
} -> i1, i1, i32
```

The number of inputs and outputs and their types must match between the first
and second circuit. All regions are isolated from above.

For more information about the `verif.lec` operation, refer to the Verif Dialect
documentation.

In the second phase, we decide to use the SMT backend for solving the LEC
statement and thus call the lowering pass to SMT and provide it the conversion
patterns necessary to encode all the operations present in the IR in terms of
SMT formulae. In our case, that consists of the already provided lowerings for
`comb`, `hw`, and `verif`. The result of this lowering is the following SMT
encoding.

```mlir
// return values indicate:
// * equivalent?
// * counter-example for "in" available?
// * counter-example value for "in" (undefined if not availbable)
// * counter-example for "two" available?
// * counter-example value for "two" (undefined if not availbable)
%0:5 = smt.solver (<non-smt-value-passthrough>)
                   <solver-options-attr> : i1, i1, i32, i1, i32 {
  // region is isolated from above
  %true = hw.constant true
  %false = hw.constant false
  %c0_i32 = hw.constant 0 : i32
  %0 = smt.declare_fun "in" : !smt.bv<32>
  %1 = smt.bv.constant #smt.bv<2> : !smt.bv<32>
  %2 = smt.bv.mul %0, %1 : !smt.bv<32>
  %3 = smt.bv.add %0, %0 : !smt.bv<32>
  %4 = smt.distinct %2, %3 : !smt.bv<32>
  smt.assert %4 // operand type is !smt.bool
  %5:3 = smt.check (<optional-assumptions>) sat {
    %6:2 = smt.eval %0 : (!smt.bv<32>) -> (i1, i32)
    %7:2 = smt.eval %1 : (!smt.bv<32>) -> (i1, i32)
    smt.yield %false, %6#0, %6#1, %7#0, %7#1 : i1, i1, i32, i1, i32
  } unknown {
    // could request a message on why it is unknown from the solver
    smt.yield %false, %false, %c0_i32, %false, %c0_i32 : i1, i1, i32, i1, i32
  } unsat {
    // could request a proof from the solver here
    smt.yield %true, %false, %c0_i32, %false, %c0_i32 : i1, i1, i32, i1, i32
  } -> i1, i1, i32, i1, i32
  smt.yield %5#0, %5#1, %5#2, %5#3, %5#4 : i1, i1, i32, i1, i32
  // no smt values may escape here
}
dbg.variable "in", %0#2 : i32
dbg.variable "two", %0#4 : i32
// TODO: how to encode that the debug variables are only
// conditionally available?
// Logic to report the result and potential counter-example
// to the user goes here (potentially using the debug
// dialect operations)
```

This SMT IR can then be lowered through one of the available backends (e.g.,
SMT-LIB, LLVM IR). Note that the SMT-LIB exporter as some considerable
restrictions on the kind of SMT IR it accepts. The ones relevant for the above
example are:
* When exporting to SMT-LIB the `smt.check` must not return any values and all
  regions must be empty. Usually, the solver prints `sat`, `unknown`, or `unsat`
  to indicate the result.
* The `smt.solver` operation must not have any return values and no passthrough
  arguments
* Optional assumptions in `smt.check` must be empty

For more information, refer to the SMT Dialect documentation.

### (Draft) Building a BMC tool

For completeness, this section describes how a BMC (bounded model checking)
compilation flow could look like using this intrastructure.
**Note:** this is just a preliminary draft for now

The flow for model checking is very similar to the LEC flow above, just using
`verif.bmc` instead of `verif.lec`. Obviously, the `seq` dialect is treated
vastly different, but `comb` and `hw` are handled the same way. Custom SMT
encodings can take advantage of this to implement both a LEC and BMC tool with
only one SMT encoding or provide separate lowering passes if necessary (or
preferred).

Let's consider the following hardware module that implements an accumulator
which only takes even numbers as input, the output of the module is the inverted
state of the accumulator register, therefore it should always be odd.

```mlir
hw.module @mod(in %clk: !seq.clock, in %arg0: i32, in %rst: i1,
               in %rst_val: i32, out out: i32) {
  %c-1_i32 = hw.constant -1 : i32
  %c0_i32 = hw.constant 0 : i32
  %c2_i32 = hw.constant 2 : i32
  %0 = comb.modu %arg0, %c2_i32 : i32
  %1 = comb.icmp eq, %0, %c0_i32 : i32
  verif.assume %1 : i1
  %2 = comb.modu %rst_val, %c2_i32 : i32
  %3 = comb.icmp eq, %2, %c0_i32 : i32
  verif.assume %3 : i1
  %4 = comb.xor %rst, %true : i1
  %5 = ltl.delay %4, 2 : i1
  %6 = ltl.concat %rst, %5 : i1, !ltl.sequence
  %7 = ltl.clock %6, posedge %clk : !ltl.sequence
  verif.assume %7 : !ltl.sequence
  %state0 = seq.compreg %8, %clk reset %rst, %rst_val : i32
  %8 = comb.add %arg0, %state0 : i32
  %9 = comb.xor %state0, %c-1_i32 : i32
  %10 = comb.modu %9, %c2_i32 : i32
  %11 = comb.icmp eq, %10, %c0_i32 : i32
  verif.assert %11
  hw.output %9 : i32
}
```

This could be lowered to a dedicated BMC operation like the following. Note that
the register was removed and instead the register state is added as an
additional input and output representing the old and the new state respectively.

```mlir
verif.bmc bound 10 {
^bb0(%clk: i1, %arg0: i32, %rst: i1, %rst_val: i32, %state0: i32):
  %true = hw.constant true
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %c2_i32 = hw.constant 2 : i32
  %c-1_i32 = hw.constant -1 : i32
  %0 = comb.modu %arg0, %c2_i32 : i32
  %1 = comb.icmp eq, %0, %c0_i32 : i32
  verif.assume %1 : i1
  %2 = comb.modu %rst_val, %c2_i32 : i32
  %3 = comb.icmp eq, %2, %c0_i32 : i32
  verif.assume %3 : i1
  %4 = comb.xor %rst, %true : i1
  %5 = ltl.delay %4, 2 : i1
  %6 = ltl.concat %rst, %5 : i1, !ltl.sequence
  %7 = ltl.clock %6, posedge %clk : !ltl.sequence
  verif.assume %7 : !ltl.sequence
  %8 = comb.add %arg0, %state0 : i32
  %9 = comb.xor %state0, %c-1_i32 : i32
  %10 = comb.modu %9, %c2_i32 : i32
  %11 = comb.icmp eq, %10, %c1_i32 : i32
  verif.assert %11
  %12 = comb.mux %rst, %rst_val, %8 : i32
  verif.yield %9, %12 : i32, i32
}
```

This CIRCT Core level representation of a BMC task can then be lowered to the
SMT dialect like the following.

```mlir
func.func @helper(%cycle: i32, %prev_clk: !smt.bv<1>, %clk: !smt.bv<1>,
                  %state0: !smt.bv<32>) -> !smt.bv<32> {
  %zero = smt.bv.constant #smt.bv<0> : !smt.bv<32>
  %one = smt.bv.constant #smt.bv<1> : !smt.bv<32>
  %c-1_i32 = smt.bv.constant #smt.bv<-1> : !smt.bv<32>
  %two = smt.bv.constant #smt.bv<2> : !smt.bv<32>
  %arg0 = smt.declare_fun "arg0" : !smt.bv<32>
  %rst = smt.declare_fun "rst" : !smt.bv<1>
  %rst_val = smt.declare_fun "rst_val" : !smt.bv<32>
  %smt_cycle = smt.from_concrete %cycle : i32 -> !smt.bv<32>
  %true = smt.bv.constant #smt.bv<1> : !smt.bv<1>
  %false = smt.bv.constant #smt.bv<0> : !smt.bv<1>
  // assumptions for this cycle 
  %2 = smt.bv.urem %arg0, %two : !smt.bv<32>
  %3 = smt.eq %2, %zero : !smt.bv<32>
  %4 = smt.bv.urem %rst_val, %two : !smt.bv<32>
  %5 = smt.eq %4, %zero : !smt.bv<32>
  %6 = smt.bv.cmp ult, %smt_cycle, %two : !smt.bv<32>
  %7 = smt.bv.cmp uge, %smt_cycle, %two : !smt.bv<32>
  %8 = smt.eq %rst, %true : !smt.bv<1>
  %9 = smt.and %6, %8
  %10 = smt.eq %rst, %false : !smt.bv<1>
  %11 = smt.and %7, %10
  %12 = smt.or %9, %11
  %assumption = smt.and %3, %5, %12
  // circuit lowered to smt
  %13 = smt.bv.add %arg0, %state0 : !smt.bv<32>
  %14 = smt.bv.xor %state0, %c-1_i32 : !smt.bv<32>
  %15 = smt.bv.urem %14, %two : !smt.bv<32>
  %16 = smt.eq %15, %one : !smt.bv<32>
  %17 = smt.implies %assumption, %16
  smt.assert %17
  %18 = smt.ite %rst, %rst_val, %13 : !smt.bv<32>
  %19 = smt.bv.not %prev_clk : !smt.bv<1>
  %posedge = smt.bv.and %clk, %19 : !smt.bv<1>
  %20 = smt.ite %posedge, %18, %state0 : !smt.bv<32>
  return %20 : !smt.bv<32>
}

smt.solver {
  %state0_init = smt.declare_fun "state0" : !smt.bv<32>
  scf.for %i = 0 to 10 step 1 iter_args(%state0 = %state0_init) {
    %clk = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %clk_nxt = smt.bv.constant #smt.bv<1> : !smt.bv<1>
    %new_state0 = func.call @helper(%i, %clk_nxt, %clk, %state0)
      : (i32, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>) -> !smt.bv<32>
    smt.check sat {} unknown {} unsat {}
    %next_cycle_state0 = func.call @helper(%i, %clk, %clk_nxt, %new_state0)
      : (i32, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>) -> !smt.bv<32>
    smt.check sat {} unknown {} unsat {}
    scf.yield %next_cycle_state0 : !smt.bv<32>
  }
  smt.yield
}
// Note: the logic to report counter-examples is omitted here.
// There are several ways to implement it, e.g.,
// * don't let the solver generate fresh symbol names, but unique them in a
//   principled way ourselves such that those identifiers can be used to
//   reconstruct the SMT expression to evaluate against the model
// * store the values representing the symbolic values (results of
//   declare_fun) in some array and iterate over it in the sat region
// TODO: exit the loop early
```
