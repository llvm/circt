// Config adds a custom runner alongside defaults.
// RUN: circt-test --list-runners -c %S/Inputs/config.json %s | FileCheck %s --check-prefix=ADDED
// ADDED-DAG: custom  formal
// ADDED-DAG: sby  formal

// Multiple -c options aggregate runners.
// RUN: circt-test --list-runners -c %S/Inputs/config-a.json -c %S/Inputs/config-b.json %s | FileCheck %s --check-prefix=MULTI
// MULTI-DAG: runA  formal
// MULTI-DAG: runB  simulation
// MULTI-DAG: sby  formal

// useDefaultRunners: false suppresses defaults.
// RUN: circt-test --list-runners -c %S/Inputs/config-no-defaults.json %s | FileCheck %s --check-prefix=NODEF
// NODEF: only  formal
// NODEF-NOT: sby
// NODEF-NOT: circt-bmc
// NODEF-NOT: verilator

// Config runner with same name as earlier runner replaces it.
// RUN: circt-test --list-runners -c %S/Inputs/config-dup.json %s | FileCheck %s --check-prefix=OVERWRITE
// OVERWRITE: runA  simulation  unavailable
// OVERWRITE-NOT: runA  formal

// Invalid JSON produces an error.
// RUN: not circt-test --list-runners -c %S/Inputs/config-bad.json %s 2>&1 | FileCheck %s --check-prefix=BADJSON
// BADJSON: error: could not parse config file

// Missing required field produces an error.
// RUN: not circt-test --list-runners -c %S/Inputs/config-missing.json %s 2>&1 | FileCheck %s --check-prefix=MISSING
// MISSING: error: config file {{.*}}: missing value at (root).runners[0].binary
