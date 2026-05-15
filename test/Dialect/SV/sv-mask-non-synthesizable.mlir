// RUN: circt-opt -split-input-file -sv-mask-non-synthesizable=mode=delete %s | FileCheck %s --check-prefix=DELETE
// RUN: circt-opt -split-input-file -sv-mask-non-synthesizable=mode=ifdef %s | FileCheck %s --check-prefix=IFDEF
// RUN: circt-opt -split-input-file -sv-mask-non-synthesizable=mode=pragma %s | FileCheck %s --check-prefix=PRAGMA
// Default mode is `ifdef`.
// RUN: circt-opt -split-input-file -sv-mask-non-synthesizable %s | FileCheck %s --check-prefix=IFDEF
// All modes are idempotent across re-runs.
// RUN: circt-opt -split-input-file -sv-mask-non-synthesizable=mode=delete %s | circt-opt -split-input-file -sv-mask-non-synthesizable=mode=delete | FileCheck %s --check-prefix=DELETE
// RUN: circt-opt -split-input-file -sv-mask-non-synthesizable=mode=ifdef %s | circt-opt -split-input-file -sv-mask-non-synthesizable=mode=ifdef | FileCheck %s --check-prefix=IFDEF
// RUN: circt-opt -split-input-file -sv-mask-non-synthesizable=mode=pragma %s | circt-opt -split-input-file -sv-mask-non-synthesizable=mode=pragma | FileCheck %s --check-prefix=PRAGMA
// `ifdef` mode honors a custom macro name and is idempotent under it.
// RUN: circt-opt -split-input-file -sv-mask-non-synthesizable='mode=ifdef macro=MY_GUARD' %s | FileCheck %s --check-prefix=CUSTOM
// RUN: circt-opt -split-input-file -sv-mask-non-synthesizable='mode=ifdef macro=MY_GUARD' %s | circt-opt -split-input-file -sv-mask-non-synthesizable='mode=ifdef macro=MY_GUARD' | FileCheck %s --check-prefix=CUSTOM

//===----------------------------------------------------------------------===//
// Coalesced runs and isolated matches in non-procedural regions.
//===----------------------------------------------------------------------===//

// IFDEF: sv.macro.decl @SYNTHESIS
// CUSTOM: sv.macro.decl @MY_GUARD

// DELETE-LABEL: hw.module @basic
// IFDEF-LABEL:  hw.module @basic
// PRAGMA-LABEL: hw.module @basic
// CUSTOM-LABEL: hw.module @basic
hw.module @basic(in %clock : i1, in %prop : i1) {
  // Untouched ops outside the masked set always pass through.
  // DELETE: hw.constant true
  // IFDEF:  hw.constant true
  // PRAGMA: hw.constant true
  // CUSTOM: hw.constant true
  %true = hw.constant true

  // A coalesced run of three masked ops becomes a single wrapper.
  //
  // DELETE-NOT: sv.assert.concurrent
  // DELETE-NOT: sv.assume.concurrent
  // DELETE-NOT: sv.cover.concurrent
  //
  // IFDEF:      sv.ifdef @SYNTHESIS {
  // IFDEF-NEXT: } else {
  // IFDEF-NEXT:   sv.assert.concurrent posedge %clock, %prop
  // IFDEF-NEXT:   sv.assume.concurrent posedge %clock, %prop
  // IFDEF-NEXT:   sv.cover.concurrent posedge %clock, %prop
  // IFDEF-NEXT: }
  //
  // CUSTOM:      sv.ifdef @MY_GUARD {
  // CUSTOM-NEXT: } else {
  // CUSTOM-NEXT:   sv.assert.concurrent posedge %clock, %prop
  // CUSTOM-NEXT:   sv.assume.concurrent posedge %clock, %prop
  // CUSTOM-NEXT:   sv.cover.concurrent posedge %clock, %prop
  // CUSTOM-NEXT: }
  //
  // PRAGMA:      sv.verbatim "// synthesis translate_off"
  // PRAGMA-NEXT: sv.assert.concurrent posedge %clock, %prop
  // PRAGMA-NEXT: sv.assume.concurrent posedge %clock, %prop
  // PRAGMA-NEXT: sv.cover.concurrent posedge %clock, %prop
  // PRAGMA-NEXT: sv.verbatim "// synthesis translate_on"
  sv.assert.concurrent posedge %clock, %prop
  sv.assume.concurrent posedge %clock, %prop
  sv.cover.concurrent posedge %clock, %prop

  // A non-masked op separates two independent runs, each gets its own wrapper.
  //
  // IFDEF:      hw.constant false
  // IFDEF-NEXT: sv.ifdef @SYNTHESIS {
  // IFDEF-NEXT: } else {
  // IFDEF-NEXT:   sv.assert_property %prop : i1
  // IFDEF-NEXT:   sv.assume_property %prop : i1
  // IFDEF-NEXT:   sv.cover_property %prop : i1
  // IFDEF-NEXT: }
  //
  // PRAGMA:      hw.constant false
  // PRAGMA-NEXT: sv.verbatim "// synthesis translate_off"
  // PRAGMA-NEXT: sv.assert_property %prop : i1
  // PRAGMA-NEXT: sv.assume_property %prop : i1
  // PRAGMA-NEXT: sv.cover_property %prop : i1
  // PRAGMA-NEXT: sv.verbatim "// synthesis translate_on"
  %false = hw.constant false
  sv.assert_property %prop : i1
  sv.assume_property %prop : i1
  sv.cover_property %prop : i1
}

// -----

//===----------------------------------------------------------------------===//
// Procedural regions: use `sv.ifdef.procedural` and place verbatim ops inside.
//===----------------------------------------------------------------------===//

// IFDEF: sv.macro.decl @SYNTHESIS

// DELETE-LABEL: hw.module @procedural
// IFDEF-LABEL:  hw.module @procedural
// PRAGMA-LABEL: hw.module @procedural
hw.module @procedural(in %clock : i1, in %prop : i1) {
  // DELETE:      sv.always posedge %clock {
  // DELETE-NEXT: }
  //
  // IFDEF:      sv.always posedge %clock {
  // IFDEF-NEXT:   sv.ifdef.procedural @SYNTHESIS {
  // IFDEF-NEXT:   } else {
  // IFDEF-NEXT:     sv.assert_property %prop : i1
  // IFDEF-NEXT:     sv.assume_property %prop : i1
  // IFDEF-NEXT:   }
  // IFDEF-NEXT: }
  //
  // PRAGMA:      sv.always posedge %clock {
  // PRAGMA-NEXT:   sv.verbatim "// synthesis translate_off"
  // PRAGMA-NEXT:   sv.assert_property %prop : i1
  // PRAGMA-NEXT:   sv.assume_property %prop : i1
  // PRAGMA-NEXT:   sv.verbatim "// synthesis translate_on"
  // PRAGMA-NEXT: }
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
// `sv.ifdef @SYNTHESIS` ancestor (recursive idempotence): a nested
// `sv.ifdef.procedural @SYNTHESIS` is not added inside an `sv.always` that
// is itself inside the else region of an outer `sv.ifdef @SYNTHESIS`.
//
// `pragma` mode similarly skips ops inside an `sv.always` that is wholly
// contained between matching `// synthesis translate_off` /
// `// synthesis translate_on` `sv.verbatim` siblings.
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

// PRAGMA-LABEL: hw.module @nested_in_existing_pragma
// IFDEF-LABEL: hw.module @nested_in_existing_pragma
hw.module @nested_in_existing_pragma(in %clock : i1, in %prop : i1) {
  // PRAGMA:      sv.verbatim "// synthesis translate_off"
  // PRAGMA-NEXT: sv.always posedge %clock {
  // PRAGMA-NEXT:   sv.assert_property %prop : i1
  // PRAGMA-NEXT: }
  // PRAGMA-NEXT: sv.verbatim "// synthesis translate_on"
  sv.verbatim "// synthesis translate_off"
  sv.always posedge %clock {
    sv.assert_property %prop : i1
  }
  sv.verbatim "// synthesis translate_on"
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
// `ifdef` mode when nothing was wrapped, no verbatim added in `pragma` mode).
//===----------------------------------------------------------------------===//

// IFDEF-NOT:    sv.macro.decl @SYNTHESIS
// IFDEF-LABEL:  hw.module @nothing_to_mask
// DELETE-LABEL: hw.module @nothing_to_mask
// PRAGMA-LABEL: hw.module @nothing_to_mask
hw.module @nothing_to_mask(in %clock : i1) {
  // IFDEF-NOT:  sv.ifdef
  // PRAGMA-NOT: sv.verbatim
  %true = hw.constant true
  sv.always posedge %clock {
    sv.if %true {}
  }
}
