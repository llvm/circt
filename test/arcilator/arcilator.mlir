// RUN: arcilator %s | FileCheck %s --check-prefix=LLVM
// RUN: arcilator --print-debug-info %s | FileCheck %s --check-prefix=LLVM-DEBUG

hw.module @Top(in %clock : !seq.clock, in %i0 : i4, in %i1 : i4, out out : i4) {
  %0 = comb.add %i0, %i1 : i4
  %1 = comb.xor %0, %i0 : i4
  %2 = comb.xor %0, %i1 : i4
  %foo = seq.compreg %1, %clock : i4
  %bar = seq.compreg %2, %clock : i4
  %3 = comb.mul %foo, %bar : i4
  hw.output %3 : i4
}

// LLVM: define void @Top_eval(ptr %0)
// LLVM:   add i4
// LLVM:   xor i4
// LLVM:   xor i4
// LLVM:   mul i4

// LLVM-DEBUG: define void @Top_eval(ptr %0){{.*}}!dbg
// LLVM-DEBUG:   add i4{{.*}}!dbg
// LLVM-DEBUG:   xor i4{{.*}}!dbg
// LLVM-DEBUG:   xor i4{{.*}}!dbg
// LLVM-DEBUG:   mul i4{{.*}}!dbg
