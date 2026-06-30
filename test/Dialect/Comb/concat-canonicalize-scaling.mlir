// Regression guard against quadratic scaling in comb.concat canonicalization.
// The companion Python script emits a module with N one-bit inputs, and a
// left-associative chain of N-1 binary concats (exercising concat flattening).
//
// With the old O(N^2) algorithms, N=2000 would take minutes.  With the
// stack-based flatten, it finishes in well under a second.

// RUN: %PYTHON% %S/concat-canonicalize-scaling.py    10 > %t.1.mlir
// RUN: %PYTHON% %S/concat-canonicalize-scaling.py   100 > %t.2.mlir
// RUN: %PYTHON% %S/concat-canonicalize-scaling.py   500 > %t.3.mlir
// RUN: %PYTHON% %S/concat-canonicalize-scaling.py  2000 > %t.4.mlir
// RUN: %PYTHON% %S/concat-canonicalize-scaling.py 10000 > %t.5.mlir

// RUN: circt-opt -canonicalize %t.1.mlir -o /dev/null
// RUN: circt-opt -canonicalize %t.2.mlir -o /dev/null
// RUN: circt-opt -canonicalize %t.3.mlir -o /dev/null
// RUN: circt-opt -canonicalize %t.4.mlir -o /dev/null
// RUN: circt-opt -canonicalize %t.5.mlir -o /dev/null
