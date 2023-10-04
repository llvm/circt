// RUN: arcilator %s --inline=0 --until-before=llvm-lowering | FileCheck %s
// RUN: arcilator %s | FileCheck %s --check-prefix=LLVM
// RUN: arcilator --print-debug-info %s | FileCheck %s --check-prefix=LLVM-DEBUG

// CHECK:      func.func @[[XOR_ARC:.+]](
// CHECK-NEXT:   comb.xor
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK:      func.func @[[ADD_ARC:.+]](
// CHECK-NEXT:   comb.add
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK:      func.func @[[MUL_ARC:.+]](
// CHECK-NEXT:   comb.mul
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK: func.func @Top_passthrough
// CHECK: func.func @Top_clock

// CHECK-NOT: hw.module @Top
// CHECK-LABEL: arc.model "Top" {
// CHECK-NEXT: ^bb0(%arg0: !arc.storage<8>):
hw.module @Top(in %clock : !seq.clock, in %i0 : i4, in %i1 : i4, out out : i4) {
  // CHECK: func.call @Top_passthrough(%arg0)
  // CHECK: scf.if {{%.+}} {
  // CHECK:   func.call @Top_clock(%arg0)
  // CHECK: }
  %0 = comb.add %i0, %i1 : i4
  %1 = comb.xor %0, %i0 : i4
  %2 = comb.xor %0, %i1 : i4
  %foo = seq.compreg %1, %clock : i4
  %bar = seq.compreg %2, %clock : i4
  %3 = comb.mul %foo, %bar : i4
  hw.output %3 : i4
}

// LLVM: define void @Top_passthrough(ptr %0)
// LLVM:   mul i4
// LLVM: define void @Top_clock(ptr %0)
// LLVM:   add i4
// LLVM:   xor i4
// LLVM:   xor i4
// LLVM: define void @Top_eval(ptr %0)
// LLVM:   call void @Top_passthrough(ptr %0)
// LLVM:   call void @Top_clock(ptr %0)

// LLVM-DEBUG: define void @Top_passthrough(ptr %0){{.*}}!dbg
// LLVM-DEBUG:   mul i4{{.*}}!dbg
// LLVM-DEBUG: define void @Top_clock(ptr %0){{.*}}!dbg
// LLVM-DEBUG:   add i4{{.*}}!dbg
// LLVM-DEBUG:   xor i4{{.*}}!dbg
// LLVM-DEBUG:   xor i4{{.*}}!dbg
// LLVM-DEBUG: define void @Top_eval(ptr %0){{.*}}!dbg
// LLVM-DEBUG:   call void @Top_passthrough(ptr %0){{.*}}!dbg
// LLVM-DEBUG:   call void @Top_clock(ptr %0){{.*}}!dbg
