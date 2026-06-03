// RUN: circt-opt -split-input-file -sv-mask-non-synthesizable=mode=delete %s | FileCheck %s --check-prefixes=COMMON,DELETE
// RUN: circt-opt -split-input-file -sv-mask-non-synthesizable=mode=ifdef %s | FileCheck %s --check-prefixes=COMMON,IFDEF
//
// Default mode is `ifdef`.
// RUN: circt-opt -split-input-file -sv-mask-non-synthesizable %s | FileCheck %s --check-prefixes=COMMON,IFDEF
//
// Both modes are idempotent across re-runs.
// RUN: circt-opt -split-input-file -sv-mask-non-synthesizable=mode=delete %s | circt-opt -split-input-file -sv-mask-non-synthesizable=mode=delete | FileCheck %s --check-prefixes=COMMON,DELETE
// RUN: circt-opt -split-input-file -sv-mask-non-synthesizable=mode=ifdef %s | circt-opt -split-input-file -sv-mask-non-synthesizable=mode=ifdef | FileCheck %s --check-prefixes=COMMON,IFDEF
//
// `ifdef` mode honors a custom macro name.
// RUN: circt-opt -split-input-file -sv-mask-non-synthesizable='mode=ifdef macro=MY_GUARD' %s | FileCheck %s --check-prefix=COMMON,CUSTOM

//===----------------------------------------------------------------------===//
// Each masked op is wrapped in its own `sv.ifdef`; adjacent wrappers are not
// coalesced (we rely on `hw-cleanup` to merge them downstream).
//===----------------------------------------------------------------------===//

// IFDEF: sv.macro.decl @SYNTHESIS
// CUSTOM: sv.macro.decl @MY_GUARD

// COMMON-LABEL: hw.module @basic
hw.module @basic(in %clock : i1, in %prop : i1) {
  // Untouched ops outside the masked set always pass through.
  // COMMON: hw.constant true
  %true = hw.constant true

  // Each masked op gets its own wrapper, even when they are adjacent.
  //
  // DELETE-NOT: sv.assert.concurrent
  // DELETE-NOT: sv.assume.concurrent
  // DELETE-NOT: sv.cover.concurrent
  //
  // IFDEF:      sv.ifdef @SYNTHESIS {
  // IFDEF-NEXT: } else {
  // IFDEF-NEXT:   sv.assert.concurrent posedge %clock, %prop
  // IFDEF-NEXT: }
  // IFDEF-NEXT: sv.ifdef @SYNTHESIS {
  // IFDEF-NEXT: } else {
  // IFDEF-NEXT:   sv.assume.concurrent posedge %clock, %prop
  // IFDEF-NEXT: }
  // IFDEF-NEXT: sv.ifdef @SYNTHESIS {
  // IFDEF-NEXT: } else {
  // IFDEF-NEXT:   sv.cover.concurrent posedge %clock, %prop
  // IFDEF-NEXT: }
  //
  // CUSTOM:      sv.ifdef @MY_GUARD {
  // CUSTOM-NEXT: } else {
  // CUSTOM-NEXT:   sv.assert.concurrent posedge %clock, %prop
  // CUSTOM-NEXT: }
  // CUSTOM-NEXT: sv.ifdef @MY_GUARD {
  // CUSTOM-NEXT: } else {
  // CUSTOM-NEXT:   sv.assume.concurrent posedge %clock, %prop
  // CUSTOM-NEXT: }
  // CUSTOM-NEXT: sv.ifdef @MY_GUARD {
  // CUSTOM-NEXT: } else {
  // CUSTOM-NEXT:   sv.cover.concurrent posedge %clock, %prop
  // CUSTOM-NEXT: }
  sv.assert.concurrent posedge %clock, %prop
  sv.assume.concurrent posedge %clock, %prop
  sv.cover.concurrent posedge %clock, %prop

  // A non-masked op separating masked ops doesn't change the per-op wrapping.
  //
  // IFDEF:      hw.constant false
  // IFDEF-NEXT: sv.ifdef @SYNTHESIS {
  // IFDEF-NEXT: } else {
  // IFDEF-NEXT:   sv.assert_property %prop : i1
  // IFDEF-NEXT: }
  // IFDEF-NEXT: sv.ifdef @SYNTHESIS {
  // IFDEF-NEXT: } else {
  // IFDEF-NEXT:   sv.assume_property %prop : i1
  // IFDEF-NEXT: }
  // IFDEF-NEXT: sv.ifdef @SYNTHESIS {
  // IFDEF-NEXT: } else {
  // IFDEF-NEXT:   sv.cover_property %prop : i1
  // IFDEF-NEXT: }
  %false = hw.constant false
  sv.assert_property %prop : i1
  sv.assume_property %prop : i1
  sv.cover_property %prop : i1
}

// -----

//===----------------------------------------------------------------------===//
// Procedural regions: use `sv.ifdef.procedural` to wrap each masked op.
//===----------------------------------------------------------------------===//

// IFDEF: sv.macro.decl @SYNTHESIS
// COMMON-LABEL: hw.module @procedural
hw.module @procedural(in %clock : i1, in %prop : i1) {
  // DELETE:      sv.always posedge %clock {
  // DELETE-NEXT: }
  //
  // IFDEF:      sv.always posedge %clock {
  // IFDEF-NEXT:   sv.ifdef.procedural @SYNTHESIS {
  // IFDEF-NEXT:   } else {
  // IFDEF-NEXT:     sv.assert_property %prop : i1
  // IFDEF-NEXT:   }
  // IFDEF-NEXT:   sv.ifdef.procedural @SYNTHESIS {
  // IFDEF-NEXT:   } else {
  // IFDEF-NEXT:     sv.assume_property %prop : i1
  // IFDEF-NEXT:   }
  // IFDEF-NEXT: }
  sv.always posedge %clock {
    sv.assert_property %prop : i1
    sv.assume_property %prop : i1
  }
}

// -----

//===----------------------------------------------------------------------===//
// `ifdef` mode reuses an existing `sv.macro.decl @SYNTHESIS` and does not
// emit a duplicate.
//===----------------------------------------------------------------------===//

// IFDEF:     sv.macro.decl @SYNTHESIS
// IFDEF-NOT: sv.macro.decl @SYNTHESIS
sv.macro.decl @SYNTHESIS

// IFDEF-LABEL: hw.module @preexisting_macro
hw.module @preexisting_macro(in %clock : i1, in %prop : i1) {
  // IFDEF:      sv.ifdef @SYNTHESIS {
  // IFDEF-NEXT: } else {
  // IFDEF-NEXT:   sv.assert.concurrent posedge %clock, %prop
  // IFDEF-NEXT: }
  sv.assert.concurrent posedge %clock, %prop
}

// -----

//===----------------------------------------------------------------------===//
// `ifdef` mode skips ops already wrapped by an enclosing
// `sv.ifdef @SYNTHESIS` ancestor (recursive idempotence): no nested
// `sv.ifdef.procedural @SYNTHESIS` is added inside an `sv.always` that
// is itself inside the else region of an outer `sv.ifdef @SYNTHESIS`.
//===----------------------------------------------------------------------===//

sv.macro.decl @SYNTHESIS

// IFDEF-LABEL: hw.module @nested_in_existing_ifdef
hw.module @nested_in_existing_ifdef(in %clock : i1, in %prop : i1) {
  // IFDEF:      sv.ifdef @SYNTHESIS {
  // IFDEF-NEXT: } else {
  // IFDEF-NEXT:   sv.always posedge %clock {
  // IFDEF-NEXT:     sv.assert_property %prop : i1
  // IFDEF-NEXT:   }
  // IFDEF-NEXT: }
  // IFDEF-NOT:  sv.ifdef.procedural
  sv.ifdef @SYNTHESIS {
  } else {
    sv.always posedge %clock {
      sv.assert_property %prop : i1
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// `ifdef` mode picks a unique symbol name when a non-`sv.macro.decl` symbol
// of the requested name already exists at top level. The `verilogName`
// attribute keeps the emitted ``ifdef`` macro spelled as the user requested.
//===----------------------------------------------------------------------===//

// IFDEF: sv.macro.decl @SYNTHESIS_0["SYNTHESIS"]

// IFDEF-LABEL: hw.module @SYNTHESIS
hw.module @SYNTHESIS() {}

// IFDEF-LABEL: hw.module @collides_with_module
hw.module @collides_with_module(in %clock : i1, in %prop : i1) {
  // IFDEF:      sv.ifdef @SYNTHESIS_0 {
  // IFDEF-NEXT: } else {
  // IFDEF-NEXT:   sv.assert.concurrent posedge %clock, %prop
  // IFDEF-NEXT: }
  sv.assert.concurrent posedge %clock, %prop
}

// -----

//===----------------------------------------------------------------------===//
// `ifdef` mode reuses an existing `sv.macro.decl` whose Verilog identifier
// matches the requested macro name, even when its sym_name differs.
//===----------------------------------------------------------------------===//

// IFDEF:     sv.macro.decl @MY_SYM["SYNTHESIS"]
// IFDEF-NOT: sv.macro.decl
sv.macro.decl @MY_SYM["SYNTHESIS"]

// IFDEF-LABEL: hw.module @reuses_existing_decl
hw.module @reuses_existing_decl(in %clock : i1, in %prop : i1) {
  // IFDEF:      sv.ifdef @MY_SYM {
  // IFDEF-NEXT: } else {
  // IFDEF-NEXT:   sv.assert.concurrent posedge %clock, %prop
  // IFDEF-NEXT: }
  sv.assert.concurrent posedge %clock, %prop
}

// -----

//===----------------------------------------------------------------------===//
// Modules with no masked ops are untouched (no `sv.macro.decl` is added in
// `ifdef` mode when nothing was wrapped).
//===----------------------------------------------------------------------===//

// IFDEF-NOT:    sv.macro.decl @SYNTHESIS
// COMMON-LABEL: hw.module @nothing_to_mask
hw.module @nothing_to_mask(in %clock : i1) {
  // IFDEF-NOT:  sv.ifdef
  %true = hw.constant true
  sv.always posedge %clock {
    sv.if %true {}
  }
}
