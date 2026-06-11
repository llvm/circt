// RUN: circt-translate --dump-di --verify-diagnostics %s | FileCheck %s

// For aggregates, per-leaf enum info lives on the inner `dbg.subfield` ops;
// the DI analysis must not collapse them into the root `DIVariable::enumDef`
// slot (that would be lossy). Consumers that need the full mapping walk
// `DIVariable::value` themselves. `--verify-diagnostics` flags any stray
// warning from the analysis.

// CHECK-LABEL: Module "MixedLeafEnums" for hw.module
// CHECK:         EnumDef #0 ("Idle"=0, "Run"=1)
// CHECK:         EnumDef #1 ("Read"=0, "Write"=1)
// CHECK:         Variable "io"
// CHECK-NOT:     enumDefRef
hw.module @MixedLeafEnums(in %state: i2, in %mode: i3, in %data: i8) {
  %stateEnum = dbg.enumdef "State", fqn "pkg.State$", {Idle = 0 : i64, Run = 1 : i64}
  %modeEnum  = dbg.enumdef "Mode",  fqn "pkg.Mode$",  {Read = 0 : i64, Write = 1 : i64}

  // Two leaves with *different* enumDefs and one plain leaf.
  %s = dbg.subfield "io.state", %state enumDef %stateEnum : i2
  %m = dbg.subfield "io.mode",  %mode  enumDef %modeEnum  : i3
  %d = dbg.subfield "io.data",  %data                     : i8

  %agg = dbg.struct {"state": %s, "mode": %m, "data": %d} : !dbg.subfield, !dbg.subfield, !dbg.subfield

  // Root dbg.variable has no enumDef; previously this path triggered a
  // graph walk that warned "enum info for non-first leaves will be lost".
  // Now it's silent; --verify-diagnostics would catch any stray warning.
  // enumDefRef must be absent for an aggregate variable.
  dbg.variable "io", %agg : !dbg.struct<!dbg.subfield, !dbg.subfield, !dbg.subfield>
}

// Scalar variable with an enumDef: DIVariable::enumDefRef must be set.
// CHECK-LABEL: Module "ScalarEnum" for hw.module
// CHECK:         EnumDef #0 ("Idle"=0, "Run"=1)
// CHECK:         Variable "state"
// CHECK:           enumDefRef = 0
hw.module @ScalarEnum(in %state: i2) {
  %e = dbg.enumdef "State", fqn "pkg.State$", {Idle = 0 : i64, Run = 1 : i64}
  dbg.variable "state", %state enumDef %e : i2
}

// Two distinct dbg.enumdef ops with identical (name, fqn, variants) must
// share one id after deduplication: only one EnumDef entry, both variables
// reference id 0.
// CHECK-LABEL: Module "DedupEnum" for hw.module
// CHECK:         EnumDef #0
// CHECK-NOT:     EnumDef #1
// CHECK:         Variable "va"
// CHECK:           enumDefRef = 0
// CHECK:         Variable "vb"
// CHECK:           enumDefRef = 0
hw.module @DedupEnum(in %a: i2, in %b: i2) {
  %e1 = dbg.enumdef "State", fqn "pkg.State$", {Idle = 0 : i64, Run = 1 : i64}
  %e2 = dbg.enumdef "State", fqn "pkg.State$", {Idle = 0 : i64, Run = 1 : i64}
  dbg.variable "va", %a enumDef %e1 : i2
  dbg.variable "vb", %b enumDef %e2 : i2
}
