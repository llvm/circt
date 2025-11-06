// RUN: circt-opt %s --hw-constprop --split-input-file | FileCheck %s

// Test basic constant propagation within a single module
// CHECK-LABEL: hw.module @BasicConstProp
hw.module @BasicConstProp(in %a: i8, out out1: i8, out out2: i8) {
  // CHECK: hw.constant
  // CHECK: hw.output
  %c5_i8 = hw.constant 5 : i8
  %c2_i8 = hw.constant 2 : i8
  %c10_i8 = hw.constant 10 : i8
  %0 = comb.mul %c5_i8, %c2_i8 : i8
  %1 = comb.add %0, %c10_i8 : i8
  %2 = comb.sub %1, %c10_i8 : i8
  hw.output %2, %c10_i8 : i8, i8
}

// -----

// Test constant propagation across module boundaries
// CHECK-LABEL: hw.module private @ConstProducer
hw.module private @ConstProducer(out out: i8) {
  // CHECK-NEXT: %c42_i8 = hw.constant 42 : i8
  // CHECK-NEXT: hw.output %c42_i8 : i8
  %c42_i8 = hw.constant 42 : i8
  hw.output %c42_i8 : i8
}

// CHECK-LABEL: hw.module @ConstConsumer
hw.module @ConstConsumer(out out: i8) {
  // CHECK-DAG: %c42_i8 = hw.constant 42 : i8
  // CHECK-DAG: hw.instance "producer" @ConstProducer
  // CHECK: hw.output %c42_i8 : i8
  %0 = hw.instance "producer" @ConstProducer() -> (out: i8)
  hw.output %0 : i8
}

// -----

// Test constant propagation through multiple levels of hierarchy
// CHECK-LABEL: hw.module private @Level2
hw.module private @Level2(out out: i16) {
  // CHECK-NEXT: %c100_i16 = hw.constant 100 : i16
  // CHECK-NEXT: hw.output %c100_i16 : i16
  %c100_i16 = hw.constant 100 : i16
  hw.output %c100_i16 : i16
}

// CHECK-LABEL: hw.module private @Level1
hw.module private @Level1(out out: i16) {
  // CHECK-DAG: %c100_i16 = hw.constant 100 : i16
  // CHECK-DAG: hw.instance "level2" @Level2
  // CHECK: hw.output %c100_i16 : i16
  %0 = hw.instance "level2" @Level2() -> (out: i16)
  hw.output %0 : i16
}

// CHECK-LABEL: hw.module @Level0
hw.module @Level0(out out: i16) {
  // CHECK-DAG: %c100_i16 = hw.constant 100 : i16
  // CHECK-DAG: hw.instance "level1" @Level1
  // CHECK: hw.output %c100_i16 : i16
  %0 = hw.instance "level1" @Level1() -> (out: i16)
  hw.output %0 : i16
}

// -----

// Test constant propagation with inputs
// CHECK-LABEL: hw.module private @AddConstant
hw.module private @AddConstant(in %x: i8, out out: i8) {
  // CHECK-DAG: %c15_i8 = hw.constant 15 : i8
  // CHECK-DAG: %c10_i8 = hw.constant 10 : i8
  // CHECK: hw.output %c15_i8 : i8
  %c10_i8 = hw.constant 10 : i8
  %0 = comb.add %x, %c10_i8 : i8
  hw.output %0 : i8
}

// CHECK-LABEL: hw.module @UseAddConstant
hw.module @UseAddConstant(out out: i8) {
  // CHECK-DAG: %c15_i8 = hw.constant 15 : i8
  // CHECK-DAG: %c5_i8 = hw.constant 5 : i8
  // CHECK-DAG: hw.instance "adder" @AddConstant
  // CHECK: hw.output %c15_i8 : i8
  %c5_i8 = hw.constant 5 : i8
  %0 = hw.instance "adder" @AddConstant(x: %c5_i8: i8) -> (out: i8)
  hw.output %0 : i8
}

// -----

// Test that public module inputs are not propagated
// CHECK-LABEL: hw.module @PublicModule
hw.module @PublicModule(in %x: i8, out out: i8) {
  // CHECK-NEXT: %c10_i8 = hw.constant 10 : i8
  // CHECK-NEXT: %[[ADD:.+]] = comb.add %x, %c10_i8
  // CHECK-NEXT: hw.output %[[ADD]] : i8
  %c10_i8 = hw.constant 10 : i8
  %0 = comb.add %x, %c10_i8 : i8
  hw.output %0 : i8
}

// -----

// Test constant propagation with multiple instances
// CHECK-LABEL: hw.module private @ConstGen
hw.module private @ConstGen(out out: i4) {
  // CHECK-NEXT: %c7_i4 = hw.constant 7 : i4
  // CHECK-NEXT: hw.output %c7_i4 : i4
  %c7_i4 = hw.constant 7 : i4
  hw.output %c7_i4 : i4
}

// CHECK-LABEL: hw.module @MultipleInstances
hw.module @MultipleInstances(out out1: i4, out out2: i4, out out3: i4) {
  // CHECK-DAG: %c7_i4 = hw.constant 7 : i4
  // CHECK-DAG: hw.instance "gen1" @ConstGen
  // CHECK-DAG: hw.instance "gen2" @ConstGen
  // CHECK-DAG: hw.instance "gen3" @ConstGen
  // CHECK: hw.output %c7_i4, %c7_i4, %c7_i4 : i4, i4, i4
  %0 = hw.instance "gen1" @ConstGen() -> (out: i4)
  %1 = hw.instance "gen2" @ConstGen() -> (out: i4)
  %2 = hw.instance "gen3" @ConstGen() -> (out: i4)
  hw.output %0, %1, %2 : i4, i4, i4
}

// -----

// Test that wires are marked as overdefined (not propagated)
// CHECK-LABEL: hw.module @WireTest
hw.module @WireTest(out out: i8) {
  // CHECK-NEXT: %c42_i8 = hw.constant 42 : i8
  // CHECK-NEXT: %wire = hw.wire %c42_i8 : i8
  // CHECK-NEXT: hw.output %wire : i8
  %c42_i8 = hw.constant 42 : i8
  %wire = hw.wire %c42_i8 : i8
  hw.output %wire : i8
}

// -----

// Test constant folding of combinational operations
// CHECK-LABEL: hw.module @CombFolding
hw.module @CombFolding(out out1: i8, out out2: i8, out out3: i8, out out4: i8) {
  // CHECK-DAG: %c15_i8 = hw.constant 15 : i8
  // CHECK-DAG: %c5_i8 = hw.constant 5 : i8
  // CHECK-DAG: %c50_i8 = hw.constant 50 : i8
  // CHECK-DAG: %c7_i8 = hw.constant 7 : i8
  // CHECK: hw.output %c15_i8, %c5_i8, %c50_i8, %c7_i8 : i8, i8, i8, i8
  %c10_i8 = hw.constant 10 : i8
  %c5_i8 = hw.constant 5 : i8
  %c15_i8 = hw.constant 15 : i8
  %c7_i8 = hw.constant 7 : i8
  %add = comb.add %c10_i8, %c5_i8 : i8
  %sub = comb.sub %c10_i8, %c5_i8 : i8
  %mul = comb.mul %c10_i8, %c5_i8 : i8
  %and = comb.and %c15_i8, %c7_i8 : i8
  hw.output %add, %sub, %mul, %and : i8, i8, i8, i8
}

// -----

// Test complex hierarchy: Diamond pattern with constant propagation
// CHECK-LABEL: hw.module private @DiamondLeaf
hw.module private @DiamondLeaf(out out: i16) {
  // CHECK-NEXT: %c255_i16 = hw.constant 255 : i16
  // CHECK-NEXT: hw.output %c255_i16 : i16
  %c255_i16 = hw.constant 255 : i16
  hw.output %c255_i16 : i16
}

// CHECK-LABEL: hw.module private @DiamondLeft
hw.module private @DiamondLeft(out out: i16) {
  // CHECK-DAG: %c255_i16 = hw.constant 255 : i16
  // CHECK-DAG: hw.instance "leaf" @DiamondLeaf
  // CHECK: hw.output %c255_i16 : i16
  %0 = hw.instance "leaf" @DiamondLeaf() -> (out: i16)
  hw.output %0 : i16
}

// CHECK-LABEL: hw.module private @DiamondRight
hw.module private @DiamondRight(out out: i16) {
  // CHECK-DAG: %c255_i16 = hw.constant 255 : i16
  // CHECK-DAG: hw.instance "leaf" @DiamondLeaf
  // CHECK: hw.output %c255_i16 : i16
  %0 = hw.instance "leaf" @DiamondLeaf() -> (out: i16)
  hw.output %0 : i16
}

// CHECK-LABEL: hw.module @DiamondTop
hw.module @DiamondTop(out out1: i16, out out2: i16, out out3: i16) {
  // CHECK-DAG: %c255_i16 = hw.constant 255 : i16
  // CHECK-DAG: %c510_i16 = hw.constant 510 : i16
  // CHECK-DAG: hw.instance "left" @DiamondLeft
  // CHECK-DAG: hw.instance "right" @DiamondRight
  // CHECK-DAG: hw.instance "direct" @DiamondLeaf
  // CHECK: hw.output %c255_i16, %c255_i16, %c510_i16 : i16, i16, i16
  %0 = hw.instance "left" @DiamondLeft() -> (out: i16)
  %1 = hw.instance "right" @DiamondRight() -> (out: i16)
  %2 = hw.instance "direct" @DiamondLeaf() -> (out: i16)
  %sum = comb.add %0, %1 : i16
  hw.output %0, %1, %sum : i16, i16, i16
}

// -----

// Test complex hierarchy: Multiple outputs with different constants
// CHECK-LABEL: hw.module private @MultiConstProducer
hw.module private @MultiConstProducer(out a: i8, out b: i8, out c: i8) {
  // CHECK-DAG: %c10_i8 = hw.constant 10 : i8
  // CHECK-DAG: %c20_i8 = hw.constant 20 : i8
  // CHECK-DAG: %c30_i8 = hw.constant 30 : i8
  // CHECK: hw.output %c10_i8, %c20_i8, %c30_i8 : i8, i8, i8
  %c10_i8 = hw.constant 10 : i8
  %c20_i8 = hw.constant 20 : i8
  %c30_i8 = hw.constant 30 : i8
  hw.output %c10_i8, %c20_i8, %c30_i8 : i8, i8, i8
}

// CHECK-LABEL: hw.module private @MultiConstCombiner
hw.module private @MultiConstCombiner(out sum: i8, out prod: i8) {
  // CHECK-DAG: %c60_i8 = hw.constant 60 : i8
  // CHECK-DAG: %c112_i8 = hw.constant 112 : i8
  // CHECK-DAG: hw.instance "producer" @MultiConstProducer
  // CHECK: hw.output %c60_i8, %c112_i8 : i8, i8
  %a, %b, %c = hw.instance "producer" @MultiConstProducer() -> (a: i8, b: i8, c: i8)
  %sum = comb.add %a, %b, %c : i8
  %prod = comb.mul %a, %b, %c : i8
  hw.output %sum, %prod : i8, i8
}

// CHECK-LABEL: hw.module @MultiConstTop
hw.module @MultiConstTop(out out1: i8, out out2: i8) {
  // CHECK-DAG: %c60_i8 = hw.constant 60 : i8
  // CHECK-DAG: %c112_i8 = hw.constant 112 : i8
  // CHECK-DAG: hw.instance "combiner" @MultiConstCombiner
  // CHECK: hw.output %c60_i8, %c112_i8 : i8, i8
  %sum, %prod = hw.instance "combiner" @MultiConstCombiner() -> (sum: i8, prod: i8)
  hw.output %sum, %prod : i8, i8
}

// -----

// Test complex hierarchy: Fan-out with multiple consumers
// CHECK-LABEL: hw.module private @FanOutSource
hw.module private @FanOutSource(out out: i8) {
  // CHECK-NEXT: %c42_i8 = hw.constant 42 : i8
  // CHECK-NEXT: hw.output %c42_i8 : i8
  %c42_i8 = hw.constant 42 : i8
  hw.output %c42_i8 : i8
}

// CHECK-LABEL: hw.module private @FanOutConsumer1
hw.module private @FanOutConsumer1(in %x: i8, out out: i8) {
  // CHECK-DAG: %c10_i8 = hw.constant 10 : i8
  // CHECK-DAG: %c52_i8 = hw.constant 52 : i8
  // CHECK: hw.output %c52_i8 : i8
  %c10_i8 = hw.constant 10 : i8
  %0 = comb.add %x, %c10_i8 : i8
  hw.output %0 : i8
}

// CHECK-LABEL: hw.module private @FanOutConsumer2
hw.module private @FanOutConsumer2(in %x: i8, out out: i8) {
  // CHECK-DAG: %c2_i8 = hw.constant 2 : i8
  // CHECK-DAG: %c84_i8 = hw.constant 84 : i8
  // CHECK: hw.output %c84_i8 : i8
  %c2_i8 = hw.constant 2 : i8
  %0 = comb.mul %x, %c2_i8 : i8
  hw.output %0 : i8
}

// CHECK-LABEL: hw.module private @FanOutConsumer3
hw.module private @FanOutConsumer3(in %x: i8, out out: i8) {
  // CHECK-DAG: %c5_i8 = hw.constant 5 : i8
  // CHECK-DAG: %c37_i8 = hw.constant 37 : i8
  // CHECK: hw.output %c37_i8 : i8
  %c5_i8 = hw.constant 5 : i8
  %0 = comb.sub %x, %c5_i8 : i8
  hw.output %0 : i8
}

// CHECK-LABEL: hw.module @FanOutTop
hw.module @FanOutTop(out out1: i8, out out2: i8, out out3: i8) {
  // CHECK-DAG: %c52_i8 = hw.constant 52 : i8
  // CHECK-DAG: %c84_i8 = hw.constant 84 : i8
  // CHECK-DAG: %c37_i8 = hw.constant 37 : i8
  // CHECK-DAG: %c42_i8 = hw.constant 42 : i8
  // CHECK-DAG: hw.instance "source" @FanOutSource
  // CHECK-DAG: hw.instance "consumer1" @FanOutConsumer1
  // CHECK-DAG: hw.instance "consumer2" @FanOutConsumer2
  // CHECK-DAG: hw.instance "consumer3" @FanOutConsumer3
  // CHECK: hw.output %c52_i8, %c84_i8, %c37_i8 : i8, i8, i8
  %source = hw.instance "source" @FanOutSource() -> (out: i8)
  %out1 = hw.instance "consumer1" @FanOutConsumer1(x: %source: i8) -> (out: i8)
  %out2 = hw.instance "consumer2" @FanOutConsumer2(x: %source: i8) -> (out: i8)
  %out3 = hw.instance "consumer3" @FanOutConsumer3(x: %source: i8) -> (out: i8)
  hw.output %out1, %out2, %out3 : i8, i8, i8
}

// -----

// Test complex hierarchy: Reconvergent paths with constant folding
// CHECK-LABEL: hw.module private @ReconvergentBase
hw.module private @ReconvergentBase(out a: i8, out b: i8) {
  // CHECK-DAG: %c10_i8 = hw.constant 10 : i8
  // CHECK-DAG: %c20_i8 = hw.constant 20 : i8
  // CHECK: hw.output %c10_i8, %c20_i8 : i8, i8
  %c10_i8 = hw.constant 10 : i8
  %c20_i8 = hw.constant 20 : i8
  hw.output %c10_i8, %c20_i8 : i8, i8
}

// CHECK-LABEL: hw.module private @ReconvergentPathA
hw.module private @ReconvergentPathA(in %x: i8, out out: i8) {
  // CHECK-DAG: %c3_i8 = hw.constant 3 : i8
  // CHECK-DAG: %c30_i8 = hw.constant 30 : i8
  // CHECK: hw.output %c30_i8 : i8
  %c3_i8 = hw.constant 3 : i8
  %0 = comb.mul %x, %c3_i8 : i8
  hw.output %0 : i8
}

// CHECK-LABEL: hw.module private @ReconvergentPathB
hw.module private @ReconvergentPathB(in %x: i8, out out: i8) {
  // CHECK-DAG: %c5_i8 = hw.constant 5 : i8
  // CHECK-DAG: %c100_i8 = hw.constant 100 : i8
  // CHECK: hw.output %c100_i8 : i8
  %c5_i8 = hw.constant 5 : i8
  %0 = comb.mul %x, %c5_i8 : i8
  hw.output %0 : i8
}

// CHECK-LABEL: hw.module @ReconvergentTop
hw.module @ReconvergentTop(out out: i8) {
  // CHECK-DAG: %c-126_i8 = hw.constant -126 : i8
  // CHECK-DAG: hw.instance "base" @ReconvergentBase
  // CHECK-DAG: hw.instance "pathA" @ReconvergentPathA
  // CHECK-DAG: hw.instance "pathB" @ReconvergentPathB
  // CHECK: hw.output %c-126_i8 : i8
  %a, %b = hw.instance "base" @ReconvergentBase() -> (a: i8, b: i8)
  %pathA_out = hw.instance "pathA" @ReconvergentPathA(x: %a: i8) -> (out: i8)
  %pathB_out = hw.instance "pathB" @ReconvergentPathB(x: %b: i8) -> (out: i8)
  %result = comb.add %pathA_out, %pathB_out : i8
  hw.output %result : i8
}

// -----

// Test multiple outputs from nested module with constant propagation
// CHECK-LABEL: hw.module @MultiOutputTest
hw.module @MultiOutputTest(out sum: i32) {
  // CHECK-DAG: %c6_i32 = hw.constant 6 : i32
  // CHECK-DAG: %c1_i32 = hw.constant 1 : i32
  // CHECK-DAG: hw.instance "nested" @MultiOutputNested
  // CHECK: hw.output %c6_i32 : i32
  %one = hw.constant 1 : i32
  %a, %b, %c = hw.instance "nested" @MultiOutputNested(input: %one: i32) -> (a: i32, b: i32, c: i32)
  %sum = comb.add %a, %b, %c : i32
  hw.output %sum : i32
}

// CHECK-LABEL: hw.module private @MultiOutputNested
hw.module private @MultiOutputNested(in %input: i32, out a: i32, out b: i32, out c: i32) {
  // CHECK-DAG: %c2_i32 = hw.constant 2 : i32
  // CHECK-DAG: %c3_i32 = hw.constant 3 : i32
  // CHECK-DAG: %c1_i32 = hw.constant 1 : i32
  // CHECK: hw.output %c1_i32, %c2_i32, %c3_i32 : i32, i32, i32
  %b = hw.constant 2 : i32
  %c = hw.constant 3 : i32
  hw.output %input, %b, %c : i32, i32, i32
}

// -----

// Test cyclic dependencies with constant propagation and extract/concat
// CHECK-LABEL: hw.module private @ExtractConcatHelper
hw.module private @ExtractConcatHelper(in %a: i2, in %unknown: i1, out out: i2) {
  // CHECK-NEXT: %[[EXT:.+]] = comb.extract %a from 0 : (i2) -> i1
  // CHECK-NEXT: %[[CONCAT:.+]] = comb.concat %unknown, %[[EXT]] : i1, i1
  // CHECK-NEXT: hw.output %[[CONCAT]] : i2
  %0 = comb.extract %a from 0 : (i2) -> i1
  %1 = comb.concat %unknown, %0 : i1, i1
  hw.output %1 : i2
}

// CHECK-LABEL: hw.module @CyclicDependency
hw.module @CyclicDependency(out out: i2) {
  // CHECK: hw.constant
  // CHECK: hw.instance "a" @ExtractConcatHelper
  // CHECK: hw.instance "b" @ExtractConcatHelper
  // CHECK: hw.output
  %c0 = hw.constant 0 : i2
  %0 = hw.instance "a" @ExtractConcatHelper(a: %c0: i2, unknown: %2: i1) -> (out: i2)
  %1 = hw.instance "b" @ExtractConcatHelper(a: %3: i2, unknown: %2: i1) -> (out: i2)
  %2 = comb.extract %1 from 0 : (i2) -> i1
  %true = hw.constant true
  %3 = comb.concat %2, %true : i1, i1
  hw.output %1 : i2
}

// -----

// Test extract operation with constant input
// CHECK-LABEL: hw.module @ExtractConstant
hw.module @ExtractConstant(out out1: i1, out out2: i1, out out3: i4) {
  // CHECK-DAG: %c-86_i8 = hw.constant -86 : i8
  // CHECK-DAG: %false = hw.constant false
  // CHECK-DAG: %true = hw.constant true
  // CHECK-DAG: %c-6_i4 = hw.constant -6 : i4
  // CHECK: hw.output %false, %true, %c-6_i4 : i1, i1, i4
  %c = hw.constant 0xAA : i8  // 10101010 in binary
  %bit0 = comb.extract %c from 0 : (i8) -> i1
  %bit7 = comb.extract %c from 7 : (i8) -> i1
  %nibble = comb.extract %c from 4 : (i8) -> i4
  hw.output %bit0, %bit7, %nibble : i1, i1, i4
}

// -----

// Test concat operation with constant inputs
// CHECK-LABEL: hw.module @ConcatConstants
hw.module @ConcatConstants(out out1: i8, out out2: i16) {
  // CHECK-DAG: %c53_i8 = hw.constant 53 : i8
  // CHECK-DAG: %c3893_i16 = hw.constant 3893 : i16
  // CHECK: hw.output %c53_i8, %c3893_i16 : i8, i16
  %c3 = hw.constant 3 : i4
  %c5 = hw.constant 5 : i4
  %result1 = comb.concat %c3, %c5 : i4, i4
  %c15 = hw.constant 15 : i8
  %result2 = comb.concat %c15, %result1 : i8, i8
  hw.output %result1, %result2 : i8, i16
}

// -----

// Test mixed extract and concat with partial constants
// CHECK-LABEL: hw.module @MixedExtractConcat
hw.module @MixedExtractConcat(in %input: i8, out out: i16) {
  // CHECK: comb.extract %input
  // CHECK: comb.concat
  // CHECK: hw.output
  %c42 = hw.constant 42 : i8
  %low = comb.extract %input from 0 : (i8) -> i4
  %high = comb.extract %c42 from 4 : (i8) -> i4
  %result = comb.concat %high, %c42, %low : i4, i8, i4
  hw.output %result : i16
}

// -----

// Test constant propagation through extract across module boundaries
// CHECK-LABEL: hw.module private @ExtractProducer
hw.module private @ExtractProducer(out low: i4, out high: i4) {
  // CHECK-DAG: %c-1_i8 = hw.constant -1 : i8
  // CHECK-DAG: %c-1_i4 = hw.constant -1 : i4
  // CHECK: hw.output %c-1_i4, %c-1_i4 : i4, i4
  %c = hw.constant 0xFF : i8
  %low = comb.extract %c from 0 : (i8) -> i4
  %high = comb.extract %c from 4 : (i8) -> i4
  hw.output %low, %high : i4, i4
}

// CHECK-LABEL: hw.module @ExtractConsumer
hw.module @ExtractConsumer(out out: i8) {
  // CHECK-DAG: %c-1_i8 = hw.constant -1 : i8
  // CHECK-DAG: hw.instance "producer" @ExtractProducer
  // CHECK: hw.output %c-1_i8 : i8
  %low, %high = hw.instance "producer" @ExtractProducer() -> (low: i4, high: i4)
  %result = comb.concat %high, %low : i4, i4
  hw.output %result : i8
}
