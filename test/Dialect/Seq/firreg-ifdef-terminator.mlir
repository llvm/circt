// RUN: circt-opt %s --lower-seq-to-sv | FileCheck %s

// This test verifies that sv.ifdef.procedural blocks created during
// register randomization have proper terminators (sv.yield) and that
// operations are inserted before the terminator, not after it.

hw.module @test_ifdef_terminator(in %clk : !seq.clock, in %rst : i1, in %in : i32, out a : i32) {
  %cst0 = hw.constant 0 : i32

  // A simple register with randomization
  %rA = seq.firreg %in clock %clk sym @regA : i32

  // CHECK-LABEL: hw.module @test_ifdef_terminator
  // CHECK: %rA = sv.reg sym @regA : !hw.inout<i32>

  // Verify the initial block is created
  // CHECK: sv.ifdef @ENABLE_INITIAL_REG_ {
  // CHECK:   sv.ordered {
  // CHECK:     sv.ifdef @FIRRTL_BEFORE_INITIAL
  // CHECK:     sv.initial {

  // Verify that sv.ifdef.procedural for INIT_RANDOM_PROLOG_ has proper structure
  // CHECK:       sv.ifdef.procedural @INIT_RANDOM_PROLOG_ {
  // CHECK-NEXT:    sv.verbatim "`INIT_RANDOM_PROLOG_"
  // The ifdef.procedural block should have an implicit sv.yield terminator
  // which is not printed when there are no operands
  // CHECK-NEXT:  }

  // Verify that sv.ifdef.procedural for RANDOMIZE_REG_INIT has proper structure
  // CHECK:       sv.ifdef.procedural @RANDOMIZE_REG_INIT {
  // CHECK:         %_RANDOM = sv.logic : !hw.inout<uarray<1xi32>>
  // CHECK:         sv.for
  // Operations should be inserted before the implicit terminator
  // CHECK:         sv.bpassign %rA
  // The ifdef.procedural block should end with an implicit sv.yield
  // CHECK-NEXT:  }

  // CHECK:     }
  // CHECK:     sv.ifdef @FIRRTL_AFTER_INITIAL
  // CHECK:   }
  // CHECK: }

  hw.output %rA : i32
}

// Test with multiple registers in the same ifdef block
hw.module @test_multiple_regs(in %clk : !seq.clock, in %in : i8, out a : i8, out b : i8) {
  // Register A - will be randomized
  %rA = seq.firreg %in clock %clk sym @regA : i8

  // Register B - will also be randomized
  %rB = seq.firreg %in clock %clk sym @regB : i8

  // CHECK-LABEL: hw.module @test_multiple_regs
  // CHECK: %rA = sv.reg sym @regA : !hw.inout<i8>
  // CHECK: %rB = sv.reg sym @regB : !hw.inout<i8>

  // CHECK: sv.ifdef @ENABLE_INITIAL_REG_ {
  // CHECK:   sv.ordered {
  // CHECK:     sv.initial {
  // CHECK:       sv.ifdef.procedural @INIT_RANDOM_PROLOG_

  // Both registers should be initialized within the same ifdef block
  // CHECK:       sv.ifdef.procedural @RANDOMIZE_REG_INIT {
  // CHECK:         %_RANDOM = sv.logic
  // Both registers should have their bpassign before the terminator
  // CHECK:         sv.bpassign %rA
  // CHECK:         sv.bpassign %rB
  // All operations should be before the implicit terminator
  // CHECK-NEXT:  }

  // CHECK:     }
  // CHECK:   }
  // CHECK: }

  hw.output %rA, %rB : i8, i8
}

// Test with async reset to verify proper terminator handling
hw.module @test_async_reset(in %clk : !seq.clock, in %rst : i1, in %in : i16, out out : i16) {
  %cst0 = hw.constant 0 : i16

  // Register with async reset - this creates different initialization
  %r1 = seq.firreg %in clock %clk sym @reg1 reset async %rst, %cst0 : i16

  // CHECK-LABEL: hw.module @test_async_reset
  // CHECK: %r1 = sv.reg sym @reg1

  // CHECK: sv.ifdef @ENABLE_INITIAL_REG_ {
  // CHECK:   sv.ordered {
  // CHECK:     sv.initial {
  // CHECK:       sv.ifdef.procedural @INIT_RANDOM_PROLOG_
  // CHECK:       sv.ifdef.procedural @RANDOMIZE_REG_INIT {
  // Register should be initialized - operations inserted before terminator
  // CHECK:         sv.bpassign %r1
  // CHECK-NEXT:  }
  // After randomization, apply reset if active
  // CHECK:       sv.if %rst {
  // CHECK:         sv.bpassign %r1, %c0_i16
  // CHECK:       }
  // CHECK:     }
  // CHECK:   }
  // CHECK: }

  hw.output %r1 : i16
}


