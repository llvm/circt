// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce --test /usr/bin/env --test-arg true --include firrtl-module-swapper --max-chunks=1 %s | FileCheck %s

// Test that the ModuleSwapper reducer can replace instances of larger modules
// with instances of smaller modules that have the same port signature.

// CHECK-LABEL: firrtl.circuit "ModuleSwapperTest"
firrtl.circuit "ModuleSwapperTest" {
  // CHECK: firrtl.module private @SmallModule
  firrtl.module private @SmallModule(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    // Small module with minimal implementation
    firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK: firrtl.module private @LargeModule
  firrtl.module private @LargeModule(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    // Large module with more complex implementation (same interface as SmallModule)
    %wire1 = firrtl.wire : !firrtl.uint<1>
    %wire2 = firrtl.wire : !firrtl.uint<1>
    %wire3 = firrtl.wire : !firrtl.uint<1>
    firrtl.connect %wire1, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %wire2, %wire1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %wire3, %wire2 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %b, %wire3 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK: firrtl.module private @AnotherLargeModule
  firrtl.module private @AnotherLargeModule(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    // Another large module with same interface but different implementation
    %wire1 = firrtl.wire : !firrtl.uint<1>
    %wire2 = firrtl.wire : !firrtl.uint<1>
    %wire3 = firrtl.wire : !firrtl.uint<1>
    %wire4 = firrtl.wire : !firrtl.uint<1>
    firrtl.connect %wire1, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %wire2, %wire1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %wire3, %wire2 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %wire4, %wire3 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %b, %wire4 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // Module with different interface - should not be affected
  // CHECK: firrtl.module private @DifferentInterface
  firrtl.module private @DifferentInterface(in %x: !firrtl.uint<2>, out %y: !firrtl.uint<2>) {
    %wire = firrtl.wire : !firrtl.uint<2>
    firrtl.connect %wire, %x : !firrtl.uint<2>, !firrtl.uint<2>
    firrtl.connect %y, %wire : !firrtl.uint<2>, !firrtl.uint<2>
  }

  // CHECK: firrtl.module @ModuleSwapperTest
  firrtl.module @ModuleSwapperTest(in %clk: !firrtl.clock, in %input: !firrtl.uint<1>, out %output1: !firrtl.uint<1>, out %output2: !firrtl.uint<1>, out %output3: !firrtl.uint<1>, out %output4: !firrtl.uint<2>) {
    // CHECK: firrtl.instance small @SmallModule
    %small_a, %small_b = firrtl.instance small @SmallModule(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)

    // CHECK: firrtl.instance large @SmallModule
    %large_a, %large_b = firrtl.instance large @LargeModule(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)

    // CHECK: firrtl.instance another @SmallModule
    %another_a, %another_b = firrtl.instance another @AnotherLargeModule(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)

    // This should remain unchanged as it has a different interface
    // CHECK: firrtl.instance diff @DifferentInterface
    %diff_x, %diff_y = firrtl.instance diff @DifferentInterface(in x: !firrtl.uint<2>, out y: !firrtl.uint<2>)

    // Connect inputs
    firrtl.connect %small_a, %input : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %large_a, %input : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %another_a, %input : !firrtl.uint<1>, !firrtl.uint<1>

    %input_ext = firrtl.pad %input, 2 : (!firrtl.uint<1>) -> !firrtl.uint<2>
    firrtl.connect %diff_x, %input_ext : !firrtl.uint<2>, !firrtl.uint<2>

    // Connect outputs
    firrtl.connect %output1, %small_b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %output2, %large_b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %output3, %another_b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %output4, %diff_y : !firrtl.uint<2>, !firrtl.uint<2>
  }
}

// Test with modules that have different port directions but same types
// CHECK-LABEL: firrtl.circuit "DirectionTest"
firrtl.circuit "DirectionTest" {
  // CHECK: firrtl.module private @SimpleInOut
  firrtl.module private @SimpleInOut(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    firrtl.connect %out, %in : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK: firrtl.module private @ComplexInOut
  firrtl.module private @ComplexInOut(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %w1 = firrtl.wire : !firrtl.uint<1>
    %w2 = firrtl.wire : !firrtl.uint<1>
    %w3 = firrtl.wire : !firrtl.uint<1>
    firrtl.connect %w1, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %w2, %w1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %w3, %w2 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %w3 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // Different direction - should not be grouped with above modules
  // CHECK: firrtl.module private @OutIn
  firrtl.module private @OutIn(out %out: !firrtl.uint<1>, in %in: !firrtl.uint<1>) {
    %wire = firrtl.wire : !firrtl.uint<1>
    firrtl.connect %wire, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %wire : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK: firrtl.module @DirectionTest
  firrtl.module @DirectionTest(in %input: !firrtl.uint<1>, out %output1: !firrtl.uint<1>, out %output2: !firrtl.uint<1>, out %output3: !firrtl.uint<1>) {
    // CHECK: firrtl.instance simple @SimpleInOut
    %simple_in, %simple_out = firrtl.instance simple @SimpleInOut(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)

    // CHECK: firrtl.instance complex @SimpleInOut
    %complex_in, %complex_out = firrtl.instance complex @ComplexInOut(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)

    // This should remain unchanged due to different port order
    // CHECK: firrtl.instance outIn @OutIn
    %outIn_out, %outIn_in = firrtl.instance outIn @OutIn(out out: !firrtl.uint<1>, in in: !firrtl.uint<1>)

    firrtl.connect %simple_in, %input : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %complex_in, %input : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %outIn_in, %input : !firrtl.uint<1>, !firrtl.uint<1>

    firrtl.connect %output1, %simple_out : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %output2, %complex_out : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %output3, %outIn_out : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// Test that instances participating in NLAs are not swapped
// CHECK-LABEL: firrtl.circuit "NLATest"
firrtl.circuit "NLATest" {
  // NLA that references an instance
  // CHECK: hw.hierpath private @nla [@NLATest::@large, @LargeNLA]
  hw.hierpath private @nla [@NLATest::@large, @LargeNLA]

  // CHECK: firrtl.module private @SmallNLA
  firrtl.module private @SmallNLA(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    // Small simple module
    firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK: firrtl.module private @LargeNLA
  firrtl.module private @LargeNLA(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    // Large module with same interface as SmallNLA
    %wire1 = firrtl.wire : !firrtl.uint<1>
    %wire2 = firrtl.wire : !firrtl.uint<1>
    %wire3 = firrtl.wire : !firrtl.uint<1>
    %wire4 = firrtl.wire : !firrtl.uint<1>
    firrtl.connect %wire1, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %wire2, %wire1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %wire3, %wire2 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %wire4, %wire3 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %b, %wire4 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK: firrtl.module @NLATest
  firrtl.module @NLATest(in %input: !firrtl.uint<1>, out %output1: !firrtl.uint<1>, out %output2: !firrtl.uint<1>, out %output3: !firrtl.uint<1>, out %output4: !firrtl.uint<1>) {
    // This instance should remain as SmallNLA (already the smallest)
    // CHECK: firrtl.instance small @SmallNLA
    %small_a, %small_b = firrtl.instance small @SmallNLA(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)

    // This instance should NOT be swapped because it participates in an NLA
    // CHECK: firrtl.instance large sym @large {annotations = [{circt.nonlocal = @nla, class = "test"}]} @LargeNLA
    %large_a, %large_b = firrtl.instance large sym @large {annotations = [{circt.nonlocal = @nla, class = "test"}]} @LargeNLA(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)

    // This instance should be swapped because it does not participate in an NLA
    // CHECK: firrtl.instance another @SmallNLA
    %another_a, %another_b = firrtl.instance another @LargeNLA(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)

    firrtl.connect %small_a, %input : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %large_a, %input : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %another_a, %input : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %output1, %small_b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %output2, %large_b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %output3, %another_b : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// Test that modules with different port names but same types are swapped
// CHECK-LABEL: firrtl.circuit "PortNameTest"
firrtl.circuit "PortNameTest" {
  // CHECK: firrtl.module private @SmallPortNames
  firrtl.module private @SmallPortNames(in %input: !firrtl.uint<1>, out %output: !firrtl.uint<1>) {
    // Small module with simple port names
    firrtl.connect %output, %input : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK: firrtl.module private @LargePortNames
  firrtl.module private @LargePortNames(in %data_in: !firrtl.uint<1>, out %data_out: !firrtl.uint<1>) {
    // Large module with different port names but same types
    %wire1 = firrtl.wire : !firrtl.uint<1>
    %wire2 = firrtl.wire : !firrtl.uint<1>
    %wire3 = firrtl.wire : !firrtl.uint<1>
    firrtl.connect %wire1, %data_in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %wire2, %wire1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %wire3, %wire2 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %data_out, %wire3 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK: firrtl.module @PortNameTest
  firrtl.module @PortNameTest(in %clk: !firrtl.clock, in %input: !firrtl.uint<1>, out %output1: !firrtl.uint<1>, out %output2: !firrtl.uint<1>) {
    // CHECK: firrtl.instance small @SmallPortNames
    %small_input, %small_output = firrtl.instance small @SmallPortNames(in input: !firrtl.uint<1>, out output: !firrtl.uint<1>)

    // This should be swapped to SmallPortNames and port names should be updated
    // CHECK: firrtl.instance large @SmallPortNames(in input: !firrtl.uint<1>, out output: !firrtl.uint<1>)
    %large_data_in, %large_data_out = firrtl.instance large @LargePortNames(in data_in: !firrtl.uint<1>, out data_out: !firrtl.uint<1>)

    // Connect inputs
    firrtl.connect %small_input, %input : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %large_data_in, %input : !firrtl.uint<1>, !firrtl.uint<1>

    // Connect outputs
    firrtl.connect %output1, %small_output : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %output2, %large_data_out : !firrtl.uint<1>, !firrtl.uint<1>
  }
}
