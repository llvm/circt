// RUN: circt-opt -lower-seq-to-sv %s | FileCheck %s
// RUN: circt-opt -lower-seq-to-sv=emit-preset-as-inline-init=false %s | FileCheck %s --check-prefix=LEGACY

// A simple preset register is emitted as an inline `sv.reg` initializer with no
// `initial` block and no ifdef guard.
// CHECK-LABEL: hw.module @Preset
// CHECK: sv.reg init %c5_i8 : !hw.inout<i8>
// CHECK-NOT: sv.initial

// With the inline-init option disabled, the preset falls back to the guarded
// `initial` block.
// LEGACY-LABEL: hw.module @Preset
// LEGACY: sv.reg : !hw.inout<i8>
// LEGACY: sv.ifdef @ENABLE_INITIAL_REG_
// LEGACY: sv.initial
hw.module @Preset(in %clock : !seq.clock, in %next : i8) {
  %r = seq.firreg %next clock %clock preset 5 : i8
}

// A regreset with a preset emits inline init plus the clocked reset logic.
// CHECK-LABEL: hw.module @PresetReset
// CHECK: sv.reg init %c0_i8 : !hw.inout<i8>

// With the inline-init option disabled, the preset falls back to the guarded
// `initial` block (reset remains clocked).
// LEGACY-LABEL: hw.module @PresetReset
// LEGACY: sv.reg : !hw.inout<i8>
// LEGACY: sv.ifdef @ENABLE_INITIAL_REG_
// LEGACY: sv.initial
hw.module @PresetReset(in %clock : !seq.clock, in %reset : i1, in %next : i8) {
  %s = seq.firreg %next clock %clock reset sync %reset, %next preset 0 : i8
}

sv.macro.decl @MyMacro

// A preset register buried under an `sv.ifdef` (guarded only by ifdefs) is
// dominance-safe, so it is emitted as an inline `sv.reg init` inside that same
// ifdef with no `initial` block and no XMR.
// CHECK-LABEL: hw.module @PresetBuried
// CHECK: sv.ifdef @MyMacro {
// CHECK:   sv.reg init %c5_i8 sym @{{.+}} : !hw.inout<i8>
// CHECK: }
// CHECK-NOT: sv.xmr.ref
// CHECK-NOT: sv.initial

// With inline-init disabled, the buried preset falls back to the XMR-based
// guarded `initial` block.
// LEGACY: hw.hierpath @[[buried_path:.+]] [@PresetBuried::@{{.+}}]
// LEGACY-LABEL: hw.module @PresetBuried
// LEGACY: sv.ifdef @MyMacro {
// LEGACY:   sv.reg sym @{{.+}} : !hw.inout<i8>
// LEGACY: }
// LEGACY: sv.initial
// LEGACY: sv.xmr.ref @[[buried_path]]
hw.module @PresetBuried(in %clock : !seq.clock, in %next : i8) {
  sv.ifdef @MyMacro {
    %r = seq.firreg %next clock %clock preset 5 : i8
  }
}

