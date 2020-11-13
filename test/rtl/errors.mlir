// RUN: circt-opt %s -split-input-file -verify-diagnostics

func @test_constant() -> i32 {
  // expected-error @+1 {{firrtl.constant attribute bitwidth doesn't match return type}}
  %a = rtl.constant(42 : i12) : i32
  return %a : i32
}

// -----

func @test_extend(%arg0: i4) -> i4 {
  // expected-error @+1 {{extension must increase bitwidth of operand}}
  %a = rtl.sext %arg0 : (i4) -> i4
  return %a : i4
}

// -----

func @test_extract(%arg0: i4) {
  // expected-error @+1 {{'rtl.extract' op from bit too large for input}}
  %a = rtl.extract %arg0 from 6 : (i4) -> i3
}

// -----

func @test_extract(%arg0: i4) {
  // expected-error @+1 {{'rtl.extract' op from bit too large for input}}
  %b = rtl.extract %arg0 from 2 : (i4) -> i3
}

// -----

func @test_and() {
  // expected-error @+1 {{'rtl.and' op expected 1 or more operands}}
  %b = rtl.and : i111
}

// -----

func @notModule () {}

rtl.module @A(%arg0: i1) {
  // expected-error @+1 {{Symbol resolved to 'func' which is not a RTL[Ext]ModuleOp}}
  rtl.instance "foo" @notModule(%arg0) : (i1) -> ()
}

// -----

rtl.module @A(%arg0: i1) {
  // expected-error @+1 {{Cannot find module definition 'doesNotExist'}}
  rtl.instance "b1" @doesNotExist(%arg0) : (i1) -> ()
}

// -----

// expected-error @+1 {{'rtl.output' op must have same number of operands as region results}}
rtl.module @A() -> (i1) { }

// -----

rtl.module @A () {}

rtl.module @B() {
  // expected-error @+1 {{has unknown extmodule parameter value 'width' = @Foo}}
  rtl.instance "foo" @A() { parameters = { width = @Foo } }: () -> ()
}