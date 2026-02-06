# Logical Equivalence Checking

**circt-lec** \[_options_] \[_mlirfile1_] \[_mlirfile2_] --c1=\[_module1_] --c2=\[_module2_]

The `circt-lec` tool takes one or two MLIR input files with operations of the
HW, Comb, and Seq dialects and the names of two modules to check for
equivalence. To get to know the exact CLI command type `circt-lec --help`.

By default, `circt-lec` will run the Z3 SMT solver (if installed), and confirm
whether the designs are equivalent. Alternatively, you can specify different
output formats to work with other solvers capable of proving the equivalence.

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

## Errors
`JIT session error: Symbols not found` indicates that Z3 could not be found on
the system. Please install and/or pass a pointer to Z3's shared library 
(`libz3.so.4`):
`circt-lec <input-options> --shared-libs=<path-to-libz3.so.4>` 

## Building a custom LEC tool

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

Consider the code example from the above. In
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
