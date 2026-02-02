# Bounded Model Checking

**circt-bmc** \[_options_] \[_mlirfile1_] -b \[_cyclecount_] --module=\[_name_]

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

## (Draft) Building a BMC tool

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