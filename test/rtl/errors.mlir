// RUN: circt-opt %s -split-input-file -verify-diagnostics

func private @test_constant() -> i32 {
  // expected-error @+1 {{firrtl.constant attribute bitwidth doesn't match return type}}
  %a = rtl.constant(42 : i12) : i32
  return %a : i32
}

// -----

func private @test_extend(%arg0: i4) -> i4 {
  // expected-error @+1 {{extension must increase bitwidth of operand}}
  %a = rtl.sext %arg0 : (i4) -> i4
  return %a : i4
}

// -----

func private @test_extract(%arg0: i4) {
  // expected-error @+1 {{'rtl.extract' op from bit too large for input}}
  %a = rtl.extract %arg0 from 6 : (i4) -> i3
}

// -----

func private @test_extract(%arg0: i4) {
  // expected-error @+1 {{'rtl.extract' op from bit too large for input}}
  %b = rtl.extract %arg0 from 2 : (i4) -> i3
}

// -----

func private @test_and() {
  // expected-error @+1 {{'rtl.and' op expected 1 or more operands}}
  %b = rtl.and : i111
}

// -----

func private @notModule () {
  return
}

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

// -----

// expected-error @+1 {{rtl.array only supports one dimension}}
func private @arrayDims(%a: !rtl.array<3 x 4 x i5>) { }

// -----

// expected-error @+1 {{invalid element for rtl.inout type}}
func private @invalidInout(%arg0: !rtl.inout<tensor<*xf32>>) { }

// -----

rtl.module @inout(%a: i42) {  // expected-note {{prior use here}}
  // expected-error @+1 {{use of value '%a' expects different type than prior uses: '<<NULL TYPE>>' vs 'i42'}}
  %aget = rtl.read_inout %a: !rtl.inout<i42>
  rtl.output %aget : i42
}
