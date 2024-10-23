# 'verif' Dialect

This dialect provides a collection of operations to express various verification concerns, such as assertions and interacting with a piece of hardware for the sake of verifying its proper functioning.

[TOC]

## Contracts

Formal contracts are a key building block of the Verif dialect to help make formal verification scale to larger designs and deep module hierarchies.
Contracts describe what a circuit expects from its inputs (`verif.require`) and what it guarantees its output to be (`verif.ensure`).
These contracts can then be used in two key ways:

1.  A contract can be _verified_ to hold by turning `require`s into `assume`s and `ensure`s into `assert`s.
    Doing so verifies that a circuit upholds the contract by placing asserts on the values it produces, and by placing assumes on the input values the circuit sees.
    In a nutshell, this checks that, assuming the inputs honor the contract, the circuit also upholds the contract.
    This check can be done very efficiently by creating `verif.formal` ops to verify each contract.

2.  A contract can be _assumed_ to hold by turning `require`s into `assert`s and `ensure`s into `assume`s.
    Doing so verifies that the inputs fed into a circuit uphold the contract, such that the outputs can be assumed to have the promised values.
    In a nutshell, this checks that the inputs to a circuit honor the contract and therefore the circuit can be assumed to uphold the contract.
    Assuming a contract can often eliminate large parts of the circuit's actual implementation, since the contracts tend to be a simpler description of a circuit's functionality.

### Multiply-by-9 Example

Consider the following example of a HW module that computes `9 * a` using a left-shift and an addition.
Note that this is using `verif.ensure_equal` as a standin for `verif.ensure(comb.icmp eq)`.
The module's output `%z` is the result of a `verif.contract` operation.
Contracts can be completely ignored by simply passing through their operands, `%1` in this case, to their results.
This is their normal interpretation outside a formal verification flow, for example for synthesis.

```mlir
hw.module @Mul9(in %a: i42, out z: i42) {
  // Compute 9*a as (a<<3)+a.
  %c3_i42 = hw.constant 3 : i42
  %0 = comb.shl %a, %c3_i42 : i42
  %1 = comb.add %a, %0 : i42

  // Contract to check that the circuit actually produces 9*a.
  %z = verif.contract %1 : i42 {
    %c9_i42 = hw.constant 9 : i42
    %a9 = comb.mul %a, %c9_i42 : i42
    verif.ensure_equal %z, %a9
    verif.yield %a9
  }

  hw.output %z : i42
}
```

Leveraging the contract in this module to simplify verification requires two steps: proving that the contract holds, and then assuming that it holds everywhere where it's used.
Proving the contract can be done by pulling it and all its fan-in cone out into a `verif.formal` op.
There it can be inlined by passing its operands through to its results and replacing ensure with assert and require with assume.
In a nutshell, passing through its operands to the results and checking the ensures verifies that the ensures hold assuming that the contract isn't there and instead the actual hardware implementation is used.
Running this formal test will prove that the `(a<<3)+a` implemented by the module is indeed the same as the `9*a` promised by the contract.

```mlir
verif.formal @Mul9_ProveContract {
  %a = verif.symbolic_value : i42

  %c3_i42 = hw.constant 3 : i42
  %0 = comb.shl %a, %c3_i42 : i42
  %1 = comb.add %a, %0 : i42

  // Contract inlined with ensure -> assert, %z -> %1 (operand)
  %c9_i42 = hw.constant 9 : i42
  %a9 = comb.mul %a, %c9_i42 : i42
  verif.assert_equal %1, %a9 : i42
}
```

Once the contract is proved, it can be assumed to hold everywhere.
Assuming it holds can be done by inlining it into its parent block and replacing its results with the operands of the `verif.yield` op in the contract body.
At the same time, ensures are replaced with assumes and requires with asserts.
Inlining the contract in this example will make the module produce the simple `9*a` from the contract as its result.
This means that the ops describing the original implementation become obsolete and will be DCEd.
And since we are replacing the contract result `%z` with its yielded value `%a9`, the generated assume `%a9 == %a9` is also a no-op and can be removed.
Making bits of a module's implementation unused is _the_ key characteristic of contracts that makes them help formal verification scale.
If all of a module's behavior can be described by one or more simpler contracts, its entire original implementation would simply disappear in favor of the simpler contracts.

```mlir
hw.module @Mul9_AssumeContract(in %a: i42, out z: i42) {
  // The original implementation has become unused since the contract yielded
  // `%a9` as a replacement value for `%1`. Dead code elimination will clean
  // this up.
  %c3_i42 = hw.constant 3 : i42
  %0 = comb.shl %a, %c3_i42 : i42
  %1 = comb.add %a, %0 : i42

  // Contract inlined with ensure -> assume, %z -> %a9 (yield)
  %c9_i42 = hw.constant 9 : i42
  %a9 = comb.mul %a, %c9_i42 : i42
  verif.assume_equal %a9, %a9 : i42  // canonicalizes away as a no-op

  hw.output %a9 : i42
}

// ----- after canonicalization, CSE, and DCE ----- //

hw.module @Mul9_AssumeContract_Simplified(in %a: i42, out z: i42) {
  %c9_i42 = hw.constant 9 : i42
  %a9 = comb.mul %a, %c9_i42 : i42
  hw.output %a9 : i42
}
```

These two constructs can coexist in the IR.
A contract can be turned into a `verif.formal` proof that it holds, and inlined everywhere else to leverage the fact that the contract holds.

### Carry-Save Compressor Example

Consider the following slightly more involved example of a compression stage you would find in a carry-save adder.
This module takes 3 input values and produces 2 output values that sum up to the same value as its inputs.
However, the contract in the module does not specify which _exact_ output values the module produces.
Instead, it uses two symbolic values to express that the module can produce _any_ output values that sum up to the inputs.
In a sense, how exactly the compressor combines 3 values into 2 is left as an implementation detail that you can’t know about, but the guarantee you can work with is that the sum will be correct.
This is different from the previous example, where the contract produced an exact replacement value for the module output.
Also, note how this uses an `assume` to constrain the sum of the symbolic values instead of an `ensure` or `require`.

```mlir
// A module that takes 3 input values and produces 2 output values that sum up
// to the same value as the inputs. Instead of just using add it uses a
// bit-parallel full adder that takes each 3-tuple of bits in the 3 inputs, runs
// them through a full adder, and treats the resulting sum and carry as the 2
// corresponding bits for its 2 output values.
hw.module @CarrySaveCompress3to2(
  in %a0: i42, in %a1: i42, in %a2: i42,
  out z0: i42, out z1: i42
) {
  %c1_i42 = hw.constant 1 : i42
  %0 = comb.xor %a0, %a1, %a2 : i42  // sum bits of FA (a0^a1^a2)
  %1 = comb.and %a0, %a1 : i42
  %2 = comb.or %a0, %a1 : i42
  %3 = comb.and %2, %a2 : i42
  %4 = comb.or %1, %3 : i42          // carry bits of FA (a0&a1 | a2&(a0|a1))
  %5 = comb.shl %4, %c1_i42 : i42    // %5 = carry << 1
  // At this point, %0+%5 is the same as %a0+%a1+%a2, but without creating a
  // long ripple-carry chain.

  // Contract to check that we output _some_ two numbers that sum up to the same
  // value as the sum of the three inputs. We don't say which exact numbers.
  %z0, %z1 = verif.contract %0, %5 {
    // The contract promises that its outputs will sum up to the same value as
    // the sum of the module inputs.
    %inputSum = comb.add %a0, %a1, %a2 : i42
    %outputSum = comb.add %z0, %z1 : i42
    verif.ensure_equal %inputSum, %outputSum : i42

    // We don't want this contract to give guarantees about what the exact
    // values of its outputs are going to be. Instead, we only want to guarantee
    // that they sum up to the right number. To express this, we pick two
    // symbolic values and constrain them to sum up to that number, and yield
    // those from the contract. This says "you'll get any two numbers that sum
    // up to the sum of the inputs".
    %any0 = verif.symbolic_value : i42
    %any1 = verif.symbolic_value : i42
    %anySum = comb.add %any0, %any1 : i42
    verif.assume_equal %anySum, %inputSum : i42
    verif.yield %any0, %any1 : i42
  }

  hw.output %z0, %z1 : i42, i42
}
```

The contract can be proven by extracting it into a new `verif.formal` op alongside its entire fan-in cone.
Again we pretend that the contract doesn't exist by passing its operands `%0` and `%5`, the actual implementation, through to its results `%z0` and `%z1`.
Replacing ensures with asserts then verifies that the module's outputs do indeed sum up to the same value as the inputs.

```mlir
verif.formal @CarrySaveCompress3to2_ProveContract {
  %a0 = verif.symbolic_value : i42
  %a1 = verif.symbolic_value : i42
  %a2 = verif.symbolic_value : i42

  %c1_i42 = hw.constant 1 : i42
  %0 = comb.xor %a0, %a1, %a2 : i42
  %1 = comb.and %a0, %a1 : i42
  %2 = comb.or %a0, %a1 : i42
  %3 = comb.and %2, %a2 : i42
  %4 = comb.or %1, %3 : i42
  %5 = comb.shl %4, %c1_i42 : i42

  // Contract inlined with ensure -> assert, (%z0, %z1) -> (%0, %5).
  // Symbolic values omitted.
  %inputSum = comb.add %a0, %a1, %a2 : i42
  %outputSum = comb.add %0, %5 : i42
  verif.assert_equal %inputSum, %outputSum : i42
}
```

With the contract proven it can be assumed to hold by inlining it everywhere and replacing its results `%z0` and `%z1` with the value yielded by the body block.
Note how assuming the contract again provides the symbolic values `%any0` and `%any1` as a replacement for the actual implementation of the module, which causes the entire implementation to be DCEd.

```mlir
hw.module @CarrySaveCompress3to2_AssumeContract(
  in %a0: i42, in %a1: i42, in %a2: i42,
  out z0: i42, out z1: i42
) {
  // The original implementation has become unused since the contract yielded
  // `%any0` and `%any1` as a replacement value for `%0` and `%5`. Dead code
  // elimination will clean this up.
  %c1_i42 = hw.constant 1 : i42
  %0 = comb.xor %a0, %a1, %a2 : i42
  %1 = comb.and %a0, %a1 : i42
  %2 = comb.or %a0, %a1 : i42
  %3 = comb.and %2, %a2 : i42
  %4 = comb.or %1, %3 : i42
  %5 = comb.shl %4, %c1_i42 : i42

  // Contract inlined with ensure -> assume, (%z0, %z1) -> (%any0, %any1).
  // Note that this produces redundant assumes that CSE can collect.
  %inputSum = comb.add %a0, %a1, %a2 : i42
  %outputSum = comb.add %any0, %any1 : i42
  verif.assume_equal %inputSum, %outputSum : i42

  %any0 = verif.symbolic_value : i42
  %any1 = verif.symbolic_value : i42
  %anySum = comb.add %any0, %any1 : i42        // redundant
  verif.assume_equal %anySum, %inputSum : i42  // redundant

  hw.output %any0, %any1 : i42, i42
}

// ----- after canonicalization, CSE, and DCE ----- //

hw.module @CarrySaveCompress3to2_AssumeContract_Simplified(
  in %a0: i42, in %a1: i42, in %a2: i42,
  out z0: i42, out z1: i42
) {
  %any0 = verif.symbolic_value : i42
  %any1 = verif.symbolic_value : i42
  %inputSum = comb.add %a0, %a1, %a2 : i42
  %outputSum = comb.add %any0, %any1 : i42
  verif.assume_equal %inputSum, %outputSum : i42
  hw.output %any0, %any1 : i42, i42
}
```

### Carry-Save Adder Example

Consider the following carry-save adder built based on the compressor from the previous example.
It takes 5 input values and sums them up.
To do so, it uses multiple instances of the compressor to compress three of the input terms down to two iteratively, until only two terms are left.
The remaining two terms are then summed up with a plain old adder to get the final result.
This carry-save adder module has its own little contract which promises that the output is going to be the sum of all input terms.

```mlir
// A module that takes 5 input values and sums them up using a carry save adder.
hw.module @CarrySaveAdder5(
  in %a0: i42, in %a1: i42, in %a2: i42, in %a3: i42, in %a4: i42,
  out z: i42
) {
  // Each stage takes 3 of the terms and compresses them to 2.
  // terms: [a0, a1, a2, a3, a4]
  %b0, %b1 = hw.instance "comp0" @CarrySaveCompress3to2(a0: %a0: i42, a1: %a1: i42, a2: %a2: i42) -> (z0: i42, z1: i42)
  // terms: [b0, b1, a3, a4]
  %c0, %c1 = hw.instance "comp1" @CarrySaveCompress3to2(a0: %b1: i42, a1: %a3: i42, a2: %a4: i42) -> (z0: i42, z1: i42)
  // terms: [b0, c0, c1]
  %d0, %d1 = hw.instance "comp2" @CarrySaveCompress3to2(a0: %b0: i42, a1: %c0: i42, a2: %c1: i42) -> (z0: i42, z1: i42)
  // terms: [d0, d1]
  %e = comb.add %d0, %d1 : i42
  // terms: [e]

  // Contract to check that the output is the sum of all inputs.
  %z = verif.contract %e {
    %inputSum = comb.add %a0, %a1, %a2, %a3, %a4, %a5 : i42
    verif.ensure_equal %z, %inputSum : i42
    verif.yield %inputSum : i42
  }

  hw.output %z : i42
}
```

Proving the contract looks as follows.
Note how the formal proof can already assume that contracts inside the compressor submodule hold.
This is a neat example of recursive use of contracts, and how proving contracts in parent modules can benefit from contracts in child modules.
Instead of having to include the compressor module implementation in this test again, potentially complicating the proof, we can already use the simplified version described by the compressor's contract.
This turns the compressor instances basically into a few additions among symbolic values, which formal solvers are very good at working with.

```mlir
verif.formal @CarrySaveAdder5_ProveContract {
  %a0 = verif.symbolic_value : i42
  %a1 = verif.symbolic_value : i42
  %a2 = verif.symbolic_value : i42
  %a3 = verif.symbolic_value : i42
  %a4 = verif.symbolic_value : i42

  // The following instances are just two symbolic values each, constrained to
  // sum up to the instance inputs. This makes for a more trivial solve.
  %b0, %b1 = hw.instance "comp0" @CarrySaveCompress3to2_AssumeContract(a0: %a0: i42, a1: %a1: i42, a2: %a2: i42) -> (z0: i42, z1: i42)
  %c0, %c1 = hw.instance "comp1" @CarrySaveCompress3to2_AssumeContract(a0: %b1: i42, a1: %a3: i42, a2: %a4: i42) -> (z0: i42, z1: i42)
  %d0, %d1 = hw.instance "comp2" @CarrySaveCompress3to2_AssumeContract(a0: %b0: i42, a1: %c0: i42, a2: %c1: i42) -> (z0: i42, z1: i42)
  %e = comb.add %d0, %d1 : i42

  // Contract inlined with ensure -> assert, %z -> %e.
  %inputSum = comb.add %a0, %a1, %a2, %a3, %a4, %a5 : i42
  verif.assert_equal %e, %inputSum : i42
}
```

The contract can then be assumed to hold by inlining it into the carry-save adder as follows.

```mlir
hw.module @CarrySaveAdder5_AssumeContract(
  in %a0: i42, in %a1: i42, in %a2: i42, in %a3: i42, in %a4: i42,
  out z: i42
) {
  // The original implementation has become unused since the contract yielded
  // `%inputSum` as a replacement value for `%e`. Dead code elimination will
  // clean this up.
  %b0, %b1 = hw.instance "comp0" @CarrySaveCompress3to2_AssumeContract(a0: %a0: i42, a1: %a1: i42, a2: %a2: i42) -> (z0: i42, z1: i42)
  %c0, %c1 = hw.instance "comp1" @CarrySaveCompress3to2_AssumeContract(a0: %b1: i42, a1: %a3: i42, a2: %a4: i42) -> (z0: i42, z1: i42)
  %d0, %d1 = hw.instance "comp2" @CarrySaveCompress3to2_AssumeContract(a0: %b0: i42, a1: %c0: i42, a2: %c1: i42) -> (z0: i42, z1: i42)
  %e = comb.add %d0, %d1 : i42

  // Contract inlined with ensure -> assume, %z -> %inputSum.
  // Trivial assume(a == a), which is a noop, can be removed.
  %inputSum = comb.add %a0, %a1, %a2, %a3, %a4, %a5 : i42

  hw.output %inputSum : i42
}

// ----- after canonicalization, CSE, and DCE ----- //

hw.module @CarrySaveAdder5_AssumeContract_Simplified(
  in %a0: i42, in %a1: i42, in %a2: i42, in %a3: i42, in %a4: i42,
  out z: i42
) {
  %inputSum = comb.add %a0, %a1, %a2, %a3, %a4, %a5 : i42
  hw.output %inputSum : i42
}
```

## Operations

[include "Dialects/VerifOps.md"]
