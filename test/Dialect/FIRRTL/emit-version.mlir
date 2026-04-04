// Tests for --firrtl-version option: correct output format at different versions.

// Default (no --firrtl-version): emits latest version header.
// RUN: circt-translate --export-firrtl %s | FileCheck %s --check-prefix=LATEST

// Targeting 5.1.0 explicitly: same as default.
// RUN: circt-translate --export-firrtl --firrtl-version=5.1.0 %s | FileCheck %s --check-prefix=LATEST

// Targeting 2.0.0: version header, pre-3.0.0 syntax ('<=', 'is invalid', 'reg with').
// RUN: circt-translate --export-firrtl --firrtl-version=2.0.0 %s | FileCheck %s --check-prefix=V200

// Targeting 3.0.0: modern connect/regreset syntax; 'public' not yet available.
// RUN: circt-translate --export-firrtl --firrtl-version=3.0.0 %s | FileCheck %s --check-prefix=V300

// Targeting 4.0.0: modern syntax, public keyword present (added in 3.3.0).
// RUN: circt-translate --export-firrtl --firrtl-version=4.0.0 %s | FileCheck %s --check-prefix=V400

// Bad version format: expects a diagnostic and non-zero exit.
// RUN: not circt-translate --export-firrtl --firrtl-version=foo %s 2>&1 | FileCheck %s --check-prefix=BAD-VERSION

// Below-minimum version: expects a diagnostic and non-zero exit.
// RUN: not circt-translate --export-firrtl --firrtl-version=1.0.0 %s 2>&1 | FileCheck %s --check-prefix=BELOW-MIN

// BAD-VERSION: invalid --firrtl-version: 'foo', expected format 'major.minor.patch'
// BELOW-MIN:   1.0.0 is below the minimum supported version 2.0.0

// LATEST: FIRRTL version 5.1.0
// V200:   FIRRTL version 2.0.0
// V300:   FIRRTL version 3.0.0
// V400:   FIRRTL version 4.0.0

firrtl.circuit "Versions" {
  // Public module: keyword present on >= 3.3.0, dropped silently on older.
  // LATEST: public module Versions
  // V200:   module Versions
  // V300:   module Versions
  // V400:   public module Versions
  firrtl.module @Versions(in %clk: !firrtl.clock,
                          in %rst: !firrtl.reset,
                          in %in: !firrtl.uint<1>,
                          out %out: !firrtl.uint<1>) {
    // Connect: 'connect' keyword on >= 3.0.0; '<=' on older versions.
    // LATEST: connect out, in
    // V200:   out <= in
    // V300:   connect out, in
    // V400:   connect out, in
    firrtl.connect %out, %in : !firrtl.uint<1>, !firrtl.uint<1>

    // RegReset: 'regreset' on >= 3.0.0; 'reg ... with :' on older versions.
    // LATEST: regreset r : UInt<1>
    // V200:   reg r : UInt<1>
    // V300:   regreset r : UInt<1>
    // V400:   regreset r : UInt<1>
    %r = firrtl.regreset %clk, %rst, %in
        : !firrtl.clock, !firrtl.reset, !firrtl.uint<1>, !firrtl.uint<1>
  }
}
