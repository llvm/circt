// RUN: circt-opt %s -verify-diagnostics --lower-seq-to-sv | FileCheck %s --check-prefixes=CHECK,NEW
// RUN: circt-opt -lower-seq-to-sv=emit-preset-as-inline-init=false %s | FileCheck %s --check-prefixes=CHECK,OLD

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

// reg init tests

// A simple preset register is emitted as an inline `sv.reg` initializer with no
// `initial` block and no ifdef guard. With the inline-init option disabled, the
// preset falls back to the guarded `initial` block.
// CHECK-LABEL: hw.module @Preset
// NEW:       sv.reg init %c5_i8 : !hw.inout<i8>
// NEW-NOT:   sv.initial
// OLD:       sv.reg : !hw.inout<i8>
// OLD:       sv.ifdef @ENABLE_INITIAL_REG_
// OLD:       sv.initial
hw.module @Preset(in %clock : !seq.clock, in %next : i8) {
  %r = seq.firreg %next clock %clock preset 5 : i8
}

// A regreset with a preset emits inline init plus the clocked reset logic. With
// the inline-init option disabled, the preset falls back to the guarded
// `initial` block (reset remains clocked).
// CHECK-LABEL: hw.module @PresetReset
// NEW:       sv.reg init %c0_i8 : !hw.inout<i8>
// OLD:       sv.reg : !hw.inout<i8>
// OLD:       sv.ifdef @ENABLE_INITIAL_REG_
// OLD:       sv.initial
hw.module @PresetReset(in %clock : !seq.clock, in %reset : i1, in %next : i8) {
  %s = seq.firreg %next clock %clock reset sync %reset, %next preset 0 : i8
}

sv.macro.decl @MyMacro

// A preset register buried under an `sv.ifdef` (guarded only by ifdefs) is
// dominance-safe, so it is emitted as an inline `sv.reg init` inside that same
// ifdef with no `initial` block and no XMR. With inline-init disabled, the
// buried preset falls back to the XMR-based guarded `initial` block.
// OLD:       hw.hierpath @[[buried_path:.+]] [@PresetBuried::@{{.+}}]
// CHECK-LABEL: hw.module @PresetBuried
// CHECK:     sv.ifdef @MyMacro {
// NEW:         sv.reg init %c5_i8 sym @{{.+}} : !hw.inout<i8>
// OLD:         sv.reg sym @{{.+}} : !hw.inout<i8>
// CHECK:     }
// NEW-NOT:   sv.xmr.ref
// NEW-NOT:   sv.initial
// OLD:       sv.initial
// OLD:       sv.xmr.ref @[[buried_path]]
hw.module @PresetBuried(in %clock : !seq.clock, in %next : i8) {
  sv.ifdef @MyMacro {
    %r = seq.firreg %next clock %clock preset 5 : i8
  }
}
