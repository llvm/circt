// RUN: not circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis)' 2>&1 | FileCheck %s --check-prefix=NO-OPS
// RUN: not circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.dot})' 2>&1 | FileCheck %s --check-prefix=BAD-ARITY
// RUN: not circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.dot:x})' 2>&1 | FileCheck %s --check-prefix=BAD-ARITY
// RUN: not circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.dot:3 allowed-ops=synth.dot:3})' 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: not circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.dot:3 sat-solver=bogus})' 2>&1 | FileCheck %s --check-prefix=BAD-SOLVER

// NO-OPS: synth-exact-synthesis requires at least one 'allowed-ops=name:arity' entry
// BAD-ARITY: expected allowed exact-synthesis op in 'name:arity' form, e.g. 'synth.dot:3'
// DUPLICATE: duplicate allowed exact-synthesis op 'synth.dot:3'
// BAD-SOLVER: unsupported or unavailable SAT solver 'bogus' (expected auto, z3, or cadical)

hw.module @dummy(in %a : i1, in %b : i1, out y : i1) {
  %0 = comb.truth_table %a, %b -> [false, false, false, true]
  hw.output %0 : i1
}
