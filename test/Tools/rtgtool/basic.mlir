// RUN: rtgtool %s --seed=0 --emit-mlir | FileCheck %s --check-prefix=MLIR
// RUN: rtgtool %s --seed=0 --emit-rendered-mlir | FileCheck %s --check-prefix=RENDERED
// RUN: rtgtool %s --seed=0 --emit-asm | FileCheck %s --check-prefix=ASM

// MLIR: rtg.label
// RENDERED: rtg.label
// ASM: label_string:
%seq0 = rtg.sequence {
  %0 = rtg.label.decl "label_string" -> i32
  rtg.label %0 : i32
} -> !rtg.sequence

// MLIR: rtgtest.instr_a
// RENDERED: rtgtest.instr_a
// ASM: instr_a
%seq1 = rtg.sequence {
  %imm1 = arith.constant 4 : i32
  %imm2 = arith.constant 8 : i32
  rtgtest.instr_a %imm1, %imm2
} -> !rtg.sequence

// MLIR: rtg.select_random
// RENDERED-NOT: rtg.select_random
// RENDERED: rtgtest.instr_a
// ASM: instr_a
rtg.sequence {
  %ratio = arith.constant 1 : i32
  rtg.select_random [%seq0, %seq1]((), () : (), ()), [%ratio, %ratio] : !rtg.sequence, !rtg.sequence
} -> !rtg.sequence
