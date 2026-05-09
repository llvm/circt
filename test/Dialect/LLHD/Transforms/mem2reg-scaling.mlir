// Regression guard against cubic scaling in --llhd-mem2reg. The companion
// Python script emits a process with N signals that all get probed and driven
// across N/2 if/else diamond merges, exercising per-slot reaching-definition
// propagation at four different sizes. This file itself carries no IR; its
// sole purpose is to drive generation and verification via the RUN lines.

// RUN: %python %S/mem2reg-scaling.py   10 > %t.1.mlir
// RUN: %python %S/mem2reg-scaling.py  100 > %t.2.mlir
// RUN: %python %S/mem2reg-scaling.py  200 > %t.3.mlir
// RUN: %python %S/mem2reg-scaling.py 1000 > %t.4.mlir

// RUN: circt-opt --llhd-mem2reg %t.1.mlir -o /dev/null
// RUN: circt-opt --llhd-mem2reg %t.2.mlir -o /dev/null
// RUN: circt-opt --llhd-mem2reg %t.3.mlir -o /dev/null
// RUN: circt-opt --llhd-mem2reg %t.4.mlir -o /dev/null
