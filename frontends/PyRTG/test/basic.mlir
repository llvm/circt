// RUN: %rtgtool% %s --seed=0 --output-format=mlir | FileCheck %s --check-prefix=MLIR
// RUN: %rtgtool% %s --seed=0 --output-format=elaborated | FileCheck %s --check-prefix=ELABORATED
// RUN: %rtgtool% %s --seed=0 --output-format=asm | FileCheck %s --check-prefix=ASM

// MLIR: rtg.sequence @seq0
// MLIR-NEXT: rtg.label_decl
// MLIR-NEXT: rtg.label
// MLIR-NEXT: }
// MLIR-NEXT: rtg.test @test0
// MLIR-NEXT: rtg.get_sequence
// MLIR-NEXT: rtg.randomize_sequence
// MLIR-NEXT: rtg.embed_sequence
// MLIR-NEXT: }

// ELABORATED: rtg.test @test0
// ELABORATED-NEXT: rtg.label_decl
// ELABORATED-NEXT: rtg.label
// ELABORATED-NEXT: }

// ASM: Begin of test0
// ASM: label_string:
// ASM: End of test0
rtg.sequence @seq0() {
  %0 = rtg.label_decl "label_string"
  rtg.label local %0
}

rtg.test @test0() {
  %0 = rtg.get_sequence @seq0 : !rtg.sequence
  %1 = rtg.randomize_sequence %0
  rtg.embed_sequence %1
}
