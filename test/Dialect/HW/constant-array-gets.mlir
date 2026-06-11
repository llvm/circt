// Regression guard against quadratic scaling with many hw.array_gets with
// constant indices.
// The companion Python script emits a module which does N array_gets with
// constant indices from an array of size N.
//
// Without optimizing for constant indices, N=10000 would take minutes.

// RUN: %PYTHON% %S/constant-array-gets.py      10 > %t.1.mlir
// RUN: %PYTHON% %S/constant-array-gets.py     100 > %t.2.mlir
// RUN: %PYTHON% %S/constant-array-gets.py     500 > %t.3.mlir
// RUN: %PYTHON% %S/constant-array-gets.py    2000 > %t.4.mlir
// RUN: %PYTHON% %S/constant-array-gets.py   10000 > %t.5.mlir
// RUN: %PYTHON% %S/constant-array-gets.py  100000 > %t.6.mlir

// RUN: circt-opt -hw-aggregate-to-comb %t.1.mlir -o /dev/null
// RUN: circt-opt -hw-aggregate-to-comb %t.2.mlir -o /dev/null
// RUN: circt-opt -hw-aggregate-to-comb %t.3.mlir -o /dev/null
// RUN: circt-opt -hw-aggregate-to-comb %t.4.mlir -o /dev/null
// RUN: circt-opt -hw-aggregate-to-comb %t.5.mlir -o /dev/null
// RUN: circt-opt -hw-aggregate-to-comb %t.6.mlir -o /dev/null
