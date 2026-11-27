// RUN: arcilator %s --emit-llvm --lookup-tables=true | FileCheck %s

// Lookup tables must lower to read-only LLVM globals instead of stack allocas.

// CHECK: [[LUT:@[a-zA-Z0-9_]+]] = internal constant [256 x i8] c"{{.*}}", align 4
// CHECK-LABEL: define void @LookupTable_eval(
// CHECK-NOT: alloca
// CHECK: getelementptr i8, ptr [[LUT]]
// CHECK: load i4, ptr
// CHECK-NOT: alloca
// CHECK: ret void

hw.module @LookupTable(in %a: i4, in %b: i4, out y: i4) {
  %0 = comb.add %a, %b : i4
  %1 = comb.xor %0, %a : i4
  %2 = comb.mul %1, %b : i4
  %3 = comb.add %2, %0 : i4
  %4 = comb.xor %3, %1 : i4
  %5 = comb.mul %4, %a : i4
  %6 = comb.add %5, %b : i4
  %7 = comb.xor %6, %2 : i4
  %8 = comb.mul %7, %3 : i4
  %9 = comb.add %8, %4 : i4
  %10 = comb.xor %9, %5 : i4
  %11 = comb.mul %10, %6 : i4
  %12 = comb.add %11, %7 : i4
  %13 = comb.xor %12, %8 : i4
  %14 = comb.mul %13, %9 : i4
  %15 = comb.add %14, %10 : i4
  %16 = comb.xor %15, %11 : i4
  %17 = comb.mul %16, %12 : i4
  %18 = comb.add %17, %13 : i4
  %19 = comb.xor %18, %14 : i4
  %20 = comb.mul %19, %15 : i4
  hw.output %20 : i4
}
