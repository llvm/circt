// RUN: circt-opt %s -verify-diagnostics --lower-seq-to-sv | FileCheck %s

// CHECK-LABEL: emit.fragment @RANDOM_INIT_FRAGMENT {
// CHECK-NEXT:    sv.verbatim "// Standard header to adapt well known macros for register randomization."
// CHECK-NEXT:    sv.verbatim "\0A// RANDOM may be set to an expression that produces a 32-bit random unsigned value."
// CHECK-NEXT:    sv.ifdef  @RANDOM {
// CHECK-NEXT:    } else {
// CHECK-NEXT:      sv.macro.def @RANDOM "$random"
// CHECK-NEXT:    }
// CHECK-NEXT:    sv.verbatim "\0A// Users can define INIT_RANDOM as general code that gets injected into the\0A// initializer block for modules with registers."
// CHECK-NEXT:    sv.ifdef  @INIT_RANDOM {
// CHECK-NEXT:    } else {
// CHECK-NEXT:      sv.macro.def @INIT_RANDOM ""
// CHECK-NEXT:    }
// CHECK-NEXT:    sv.verbatim "\0A// If using random initialization, you can also define RANDOMIZE_DELAY to\0A// customize the delay used, otherwise 0.002 is used."
// CHECK-NEXT:    sv.ifdef  @RANDOMIZE_DELAY {
// CHECK-NEXT:    } else {
// CHECK-NEXT:      sv.macro.def @RANDOMIZE_DELAY "0.002"
// CHECK-NEXT:    }
// CHECK-NEXT:    sv.verbatim "\0A// Define INIT_RANDOM_PROLOG_ for use in our modules below."
// CHECK-NEXT:    sv.ifdef  @INIT_RANDOM_PROLOG_ {
// CHECK-NEXT:    } else {
// CHECK-NEXT:      sv.ifdef  @RANDOMIZE {
// CHECK-NEXT:        sv.ifdef  @VERILATOR {
// CHECK-NEXT:          sv.macro.def @INIT_RANDOM_PROLOG_ "`INIT_RANDOM"
// CHECK-NEXT:        } else {
// CHECK-NEXT:          sv.macro.def @INIT_RANDOM_PROLOG_ "`INIT_RANDOM #`RANDOMIZE_DELAY begin end"
// CHECK-NEXT:        }
// CHECK-NEXT:      } else {
// CHECK-NEXT:        sv.macro.def @INIT_RANDOM_PROLOG_ ""
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-LABEL: emit.fragment @RANDOM_INIT_MEM_FRAGMENT {
// CHECK-NEXT:    sv.verbatim "\0A// Include rmemory initializers in init blocks unless synthesis is set"
// CHECK-NEXT:    sv.ifdef  @RANDOMIZE {
// CHECK-NEXT:    } else {
// CHECK-NEXT:      sv.ifdef  @RANDOMIZE_MEM_INIT {
// CHECK-NEXT:        sv.macro.def @RANDOMIZE ""
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    sv.ifdef  @SYNTHESIS {
// CHECK-NEXT:    } else {
// CHECK-NEXT:      sv.ifdef  @ENABLE_INITIAL_MEM_ {
// CHECK-NEXT:      } else {
// CHECK-NEXT:        sv.macro.def @ENABLE_INITIAL_MEM_ ""
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    sv.verbatim ""
// CHECK-NEXT:  }
// CHECK-LABEL: emit.fragment @RANDOM_INIT_REG_FRAGMENT {
// CHECK-NEXT:    sv.verbatim "\0A// Include register initializers in init blocks unless synthesis is set"
// CHECK-NEXT:    sv.ifdef  @RANDOMIZE {
// CHECK-NEXT:    } else {
// CHECK-NEXT:      sv.ifdef  @RANDOMIZE_REG_INIT {
// CHECK-NEXT:        sv.macro.def @RANDOMIZE ""
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    sv.ifdef  @SYNTHESIS {
// CHECK-NEXT:    } else {
// CHECK-NEXT:      sv.ifdef  @ENABLE_INITIAL_REG_ {
// CHECK-NEXT:      } else {
// CHECK-NEXT:        sv.macro.def @ENABLE_INITIAL_REG_ ""
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    sv.verbatim ""
// CHECK-NEXT:  }

emit.fragment @SomeFragment {}

// CHECK-LABEL: hw.module.generated
// CHECK-SAME:    emit.fragments = [@RANDOM_INIT_REG_FRAGMENT, @RANDOM_INIT_MEM_FRAGMENT, @RANDOM_INIT_FRAGMENT]

// CHECK-LABEL: hw.module @fragment_ref(in %clk : i1)
// CHECK-SAME: emit.fragments = [@SomeFragment, @RANDOM_INIT_REG_FRAGMENT, @RANDOM_INIT_FRAGMENT]
hw.module @fragment_ref(in %clk : !seq.clock) attributes {emit.fragments = [@SomeFragment]} {
  %cst0_i32 = hw.constant 0 : i32
  %rA = seq.firreg %cst0_i32 clock %clk sym @regA : i32

  %0 = seq.firmem 0, 1, undefined, undefined : <3 x 19>
}
