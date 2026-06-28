// RUN: circt-translate --export-firrtl %s | FileCheck %s --check-prefix=FIR
// RUN: circt-translate --export-firrtl %s | circt-translate --import-firrtl | FileCheck %s
// `asReset` is only available from FIRRTL 7.0.0, and a bare `UInt<1>` does not
// re-import as a reset, so a reset constant has no earlier-version form:
// exporting one to an older target version is rejected rather than emitting
// output that will not re-parse.
// RUN: not circt-translate --export-firrtl --firrtl-version=4.0.0 %s 2>&1 | FileCheck %s --check-prefix=V4

// V4: reset constants requires FIRRTL 7.0.0

// A reset constant must export as `asReset(...)` rather than a bare `UInt<1>`,
// so that the emitted FIRRTL re-parses type-correctly under strict reset typing
// (where a `UInt<1>` is not accepted directly as a reset value).
firrtl.circuit "ResetConst" {
  // FIR: connect o, asReset(UInt<1>(0))
  // CHECK-LABEL: firrtl.module @ResetConst
  // CHECK: %[[R:.+]] = firrtl.asReset
  // CHECK: firrtl.matchingconnect %o, %{{.+}} : !firrtl.reset
  firrtl.module @ResetConst(out %o: !firrtl.reset) {
    %c = firrtl.specialconstant 0 : !firrtl.reset
    firrtl.matchingconnect %o, %c : !firrtl.reset
  }
}
