// RUN: circt-opt --pass-pipeline='builtin.module(hw.module(synth-test-priority-cuts{max-cuts-per-root=8}))' %s --mlir-disable-threading | FileCheck %s --check-prefixes=CHECK,ROOT8
// RUN: circt-opt --pass-pipeline='builtin.module(hw.module(synth-test-priority-cuts{max-cuts-per-root=2}))' %s --mlir-disable-threading | FileCheck %s --check-prefixes=CHECK,ROOT2
// RUN: circt-opt --pass-pipeline='builtin.module(hw.module(synth-test-priority-cuts{max-cuts-per-root=8 max-cut-input-size=2}))' %s --mlir-disable-threading | FileCheck %s --check-prefixes=CHECK,INPUT2

//===----------------------------------------------------------------------===//
// Cut Notation Explanation:
//
// Each cut is represented as: {inputs}@t<truthtable>d<depth>
//
// Components:
// - {inputs}: External inputs to the cut (cut boundary values)
//   Example: {a0, a1} means inputs a0 and a1 feed into this cut
//
// - @t<number>: Truth table as decimal representation of binary function
//   Example: @t8 = binary 1000 = 2-input AND function
//           @t2 = binary 10 = identity function (1-input)
//           @t128 = binary 10000000 = 3-input function
//
// - d<depth>: Logic depth from primary inputs to this cut's output
//   Example: d0 = primary input (no logic), d1 = one logic level, etc.
//
// Example: {a0, a1}@t8d1 means:
// - Cut has inputs a0 and a1
// - Implements AND function (truth table 8 = binary 1000)
// - Has logic depth 1 (one level from primary inputs)
//===----------------------------------------------------------------------===//


// CHECK-LABEL: Enumerating cuts for module: trivial
hw.module @trivial(in %a : i1, out result : i1) {
    // CHECK: a 1 cuts: {a}@t2d0
    // CHECK-NEXT: out0 2 cuts: {out0}@t2d0 {a}@t2d1
    // CHECK-NEXT: Cut enumeration completed successfully
    %out0 = aig.and_inv %a, %a {sv.namehint = "out0"} : i1
    hw.output %out0 : i1
}

// CHECK-LABEL: Enumerating cuts for module: extract
hw.module @extract(in %a : i2, out result : i1) {
    // CHECK:      a0 1 cuts: {a0}@t2d0
    // CHECK-NEXT: a1 1 cuts: {a1}@t2d0
    // CHECK-NEXT: out0 2 cuts: {out0}@t2d0 {a0, a1}@t8d1
    // CHECK-NEXT: Cut enumeration completed successfully
    %a0 = comb.extract %a from 0 {sv.namehint = "a0"}: (i2) -> i1
    %a1 = comb.extract %a from 1 {sv.namehint = "a1"}: (i2) -> i1
    %out0 = aig.and_inv %a0, %a1 {sv.namehint = "out0"} : i1
    hw.output %out0 : i1
}

// CHECK-LABEL: Enumerating cuts for module: test
hw.module @test(in %a : i4, in %b : i2, out result : i3) {
    // CHECK:     a0 1 cuts: {a0}@t2d0
    // CHECK-NEXT: a1 1 cuts: {a1}@t2d0
    // CHECK-NEXT: and0 2 cuts: {and0}@t2d0 {a0, a1}@t8d1
    // CHECK-NEXT: a2 1 cuts: {a2}@t2d0
    // CHECK-NEXT: b0 1 cuts: {b0}@t2d0
    // CHECK-NEXT: and1 2 cuts: {and1}@t2d0 {a2, b0}@t2d1
    // CHECK-NEXT: a3 1 cuts: {a3}@t2d0
    // CHECK-NEXT: b1 1 cuts: {b1}@t2d0
    // ROOT8-NEXT: and2 2 cuts: {and2}@t2d0 {a3, b1}@t8d1
    // ROOT8-NEXT: out0 3 cuts: {out0}@t2d0 {and0, b0}@t8d1 {a0, a1, b0}@t128d2
    // ROOT8-NEXT: out1 5 cuts: {out1}@t2d0 {and1, and0}@t2d1 {a0, a1, and1}@t112d2 {a2, b0, and0}@t2d2 {a2, b0, a0, a1}@t546d2
    // ROOT8-NEXT: out2 3 cuts: {out2}@t2d0 {and2, b1}@t8d1 {a3, b1}@t8d2
    // Check that the cuts are correctly limited when max-cuts-per-root is set to 2.
    // Currently (depth, input size) is used as the cut priority.
    // ROOT2:      and2 2 cuts: {and2}@t2d0 {a3, b1}@t8d1
    // ROOT2-NEXT: out0 2 cuts: {out0}@t2d0 {and0, b0}@t8d1
    // ROOT2-NEXT: out1 2 cuts: {out1}@t2d0 {and1, and0}@t2d1
    // ROOT2-NEXT: out2 2 cuts: {out2}@t2d0 {and2, b1}@t8d1
    // INPUT2:    and2 2 cuts: {and2}@t2d0 {a3, b1}@t8d1
    // INPUT2-NEXT: out0 2 cuts: {out0}@t2d0 {and0, b0}@t8d1
    // INPUT2-NEXT: out1 2 cuts: {out1}@t2d0 {and1, and0}@t2d1
    // INPUT2-NEXT: out2 3 cuts: {out2}@t2d0 {and2, b1}@t8d1 {a3, b1}@t8d2
    // CHECK-NEXT: Cut enumeration completed successfully

    // Extract individual bits from multi-bit inputs
    %a0 = comb.extract %a from 0 {sv.namehint = "a0"}: (i4) -> i1
    %a1 = comb.extract %a from 1 {sv.namehint = "a1"}: (i4) -> i1
    %a2 = comb.extract %a from 2 {sv.namehint = "a2"}: (i4) -> i1
    %a3 = comb.extract %a from 3 {sv.namehint = "a3"}: (i4) -> i1

    %b0 = comb.extract %b from 0 {sv.namehint = "b0"}: (i2) -> i1
    %b1 = comb.extract %b from 1 {sv.namehint = "b1"} : (i2) -> i1

    %and0 = aig.and_inv %a0, %a1 {sv.namehint = "and0"} : i1
    %and1 = aig.and_inv %a2, not %b0 {sv.namehint = "and1"} : i1
    %and2 = aig.and_inv %a3, %b1 {sv.namehint = "and2"} : i1

    %out0 = aig.and_inv %and0, %b0 {sv.namehint = "out0"} : i1
    %out1 = aig.and_inv %and1, not %and0 {sv.namehint = "out1"} : i1
    %out2 = aig.and_inv %and2, %b1 {sv.namehint = "out2"} : i1

    // Concatenate results into multi-bit output
    %result_concat = comb.concat %out2, %out1, %out0 {sv.namehint = "result_concat"} : i1, i1, i1

    hw.output %result_concat : i3
}
