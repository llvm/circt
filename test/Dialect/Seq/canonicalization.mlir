// RUN: circt-opt -canonicalize %s | FileCheck %s

hw.module.extern @Observe(%x: i32)

// CHECK-LABEL: @FirReg
hw.module @FirReg(%clk: i1, %in: i32) {
  %false = hw.constant false
  %true = hw.constant true

  // Registers that update to themselves should be replaced with a constant 0.
  %reg0 = seq.firreg %reg0 clock %clk : i32
  hw.instance "reg0" @Observe(x: %reg0: i32) -> ()
  // CHECK: hw.instance "reg0" @Observe(x: %c0_i32: i32) -> ()

  // Registers that are never clocked should be replaced with a constant 0.
  %reg1a = seq.firreg %in clock %false : i32
  %reg1b = seq.firreg %in clock %true : i32
  hw.instance "reg1a" @Observe(x: %reg1a: i32) -> ()
  hw.instance "reg1b" @Observe(x: %reg1b: i32) -> ()
  // CHECK: hw.instance "reg1a" @Observe(x: %c0_i32: i32) -> ()
  // CHECK: hw.instance "reg1b" @Observe(x: %c0_i32: i32) -> ()
}

// Should not optimize away the register if it has a symbol.
// CHECK-LABEL: @FirRegSymbol
hw.module @FirRegSymbol(%clk: i1) -> (out : i32) {
  // CHECK: %reg = seq.firreg %reg clock %clk sym @reg : i32
  // CHECK: hw.output %reg : i32
  %reg = seq.firreg %reg clock %clk sym @reg : i32
  hw.output %reg : i32
}

// CHECK-LABEL: @FirRegReset
hw.module @FirRegReset(%clk: i1, %in: i32, %r : i1, %v : i32) {
  %false = hw.constant false
  %true = hw.constant true

  // Registers that update to themselves should be replaced with their reset
  // value.
  %reg0 = seq.firreg %reg0 clock %clk reset sync %r, %v : i32
  hw.instance "reg0" @Observe(x: %reg0: i32) -> ()
  // CHECK: hw.instance "reg0" @Observe(x: %v: i32) -> ()

  // Registers that never reset should drop their reset value.
  %reg1 = seq.firreg %in clock %clk reset sync %false, %v : i32
  hw.instance "reg1" @Observe(x: %reg1: i32) -> ()
  // CHECK: %reg1 = seq.firreg %in clock %clk : i32
  // CHECK: hw.instance "reg1" @Observe(x: %reg1: i32) -> ()

  // Registers that are permanently reset should be replaced with their reset
  // value.
  %reg2a = seq.firreg %in clock %clk reset sync %true, %v : i32
  %reg2b = seq.firreg %in clock %clk reset async %true, %v : i32
  hw.instance "reg2a" @Observe(x: %reg2a: i32) -> ()
  hw.instance "reg2b" @Observe(x: %reg2b: i32) -> ()
  // CHECK: hw.instance "reg2a" @Observe(x: %v: i32) -> ()
  // CHECK: hw.instance "reg2b" @Observe(x: %v: i32) -> ()

  // Registers that are never clocked should be replaced with their reset value.
  %reg3a = seq.firreg %in clock %false reset sync %r, %v : i32
  %reg3b = seq.firreg %in clock %true reset sync %r, %v : i32
  hw.instance "reg3a" @Observe(x: %reg3a: i32) -> ()
  hw.instance "reg3b" @Observe(x: %reg3b: i32) -> ()
  // CHECK: hw.instance "reg3a" @Observe(x: %v: i32) -> ()
  // CHECK: hw.instance "reg3b" @Observe(x: %v: i32) -> ()
}

// CHECK-LABEL: @FirRegAggregate
hw.module @FirRegAggregate(%clk: i1) -> (out : !hw.struct<foo: i32>) {
  // TODO: Use constant aggregate attribute once supported.
  // CHECK:      %c0_i32 = hw.constant 0 : i32
  // CHECK-NEXT: %0 = hw.bitcast %c0_i32 : (i32) -> !hw.struct<foo: i32>
  // CHECK-NEXT: hw.output %0
  %reg = seq.firreg %reg clock %clk : !hw.struct<foo: i32>
  hw.output %reg : !hw.struct<foo: i32>
}

// CHECK-LABEL: @UninitializedArrayElement
hw.module @UninitializedArrayElement(%a: i1, %clock: i1) -> (b: !hw.array<2xi1>) {
  // CHECK:      %false = hw.constant false
  // CHECK-NEXT: %0 = hw.array_create %false, %a : i1
  // CHECK-NEXT: %r = seq.firreg %0 clock %clock : !hw.array<2xi1>
  // CHECK-NEXT: hw.output %r : !hw.array<2xi1>
  %true = hw.constant true
  %r = seq.firreg %1 clock %clock : !hw.array<2xi1>
  %0 = hw.array_get %r[%true] : !hw.array<2xi1>, i1
  %1 = hw.array_create %0, %a : i1
  hw.output %r : !hw.array<2xi1>
}
