// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-LABEL: llhd.entity @test1
llhd.entity @test1() -> () {
  // CHECK-NEXT: %c5_i32 = hw.constant 5 : i32
  %0 = moore.mir.constant 5 : !moore.int
  // CHECK-NEXT: %c3_i32 = hw.constant 3 : i32
  // CHECK-NEXT: [[SIG:%.*]] = llhd.sig "varname" %c3_i32 : i32
  %1 = moore.mir.vardecl "varname" = 3 : !moore.int
  // CHECK-NEXT: [[TIME:%.*]] = llhd.constant_time <0s, 0d, 1e>
  // CHECK-NEXT: llhd.drv [[SIG]], %c5_i32 after [[TIME]] : !llhd.sig<i32>
  moore.mir.assign %1, %0 : !moore.int
}

// CHECK-LABEL: func @FuncArgsAndReturns
// CHECK-SAME: (%arg0: i8, %arg1: i32, %arg2: i1) -> i8
func @FuncArgsAndReturns(%arg0: !moore.byte, %arg1: !moore.int, %arg2: !moore.bit) -> !moore.byte {
  // CHECK-NEXT: return %arg0 : i8
  return %arg0 : !moore.byte
}

// CHECK-LABEL: func @ControlFlow
// CHECK-SAME: (%arg0: i32, %arg1: i1)
func @ControlFlow(%arg0: !moore.int, %arg1: i1) {
  // CHECK-NEXT:   cf.br ^bb1(%arg0 : i32)
  // CHECK-NEXT: ^bb1(%0: i32):
  // CHECK-NEXT:   cf.cond_br %arg1, ^bb1(%0 : i32), ^bb2(%arg0 : i32)
  // CHECK-NEXT: ^bb2(%1: i32):
  // CHECK-NEXT:   return
  cf.br ^bb1(%arg0: !moore.int)
^bb1(%0: !moore.int):
  cf.cond_br %arg1, ^bb1(%0 : !moore.int), ^bb2(%arg0 : !moore.int)
^bb2(%1: !moore.int):
  return
}

// CHECK-LABEL: func @Calls
// CHECK-SAME: (%arg0: i8, %arg1: i32, %arg2: i1) -> i8
func @Calls(%arg0: !moore.byte, %arg1: !moore.int, %arg2: !moore.bit) -> !moore.byte {
  // CHECK-NEXT: %true =
  // CHECK-NEXT: call @ControlFlow(%arg1, %true) : (i32, i1) -> ()
  // CHECK-NEXT: [[TMP:%.+]] = call @FuncArgsAndReturns(%arg0, %arg1, %arg2) : (i8, i32, i1) -> i8
  // CHECK-NEXT: return [[TMP]] : i8
  %true = hw.constant true
  call @ControlFlow(%arg1, %true) : (!moore.int, i1) -> ()
  %0 = call @FuncArgsAndReturns(%arg0, %arg1, %arg2) : (!moore.byte, !moore.int, !moore.bit) -> !moore.byte
  return %0 : !moore.byte
}

// CHECK-LABEL: func @UnrealizedConversionCast
func @UnrealizedConversionCast(%arg0: !moore.byte) -> !moore.shortint {
  // CHECK-NEXT: [[TMP:%.+]] = comb.concat %arg0, %arg0 : i8, i8
  // CHECK-NEXT: return [[TMP]] : i16
  %0 = builtin.unrealized_conversion_cast %arg0 : !moore.byte to i8
  %1 = comb.concat %0, %0 : i8, i8
  %2 = builtin.unrealized_conversion_cast %1 : i16 to !moore.shortint
  return %2 : !moore.shortint
}
