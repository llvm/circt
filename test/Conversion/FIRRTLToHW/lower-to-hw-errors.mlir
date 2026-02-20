// RUN: circt-opt -lower-firrtl-to-hw=warn-on-unprocessed-annotations -verify-diagnostics -split-input-file -allow-unregistered-dialect -mlir-disable-threading %s

// The firrtl.circuit should be removed, the main module name moved to an
// attribute on the module.
// CHECK-NOT: firrtl.circuit

// We should get a large header boilerplate.
// CHECK:   sv.ifdef.procedural "RANDOMIZE_GARBAGE_ASSIGN"  {
// CHECK-NEXT:   sv.verbatim "`define RANDOMIZE"
// CHECK-NEXT:  }

firrtl.circuit "OperandTypeIsFIRRTL" {
  firrtl.module @OperandTypeIsFIRRTL() { }
    func.func @Test() {
    // expected-error @+1 {{Found unhandled FIRRTL operation 'firrtl.constant'}}
    %a = firrtl.constant 0 : !firrtl.uint<1>
    return
  }
}

// -----

firrtl.circuit "ResultTypeIsFIRRTL" {
  firrtl.module @ResultTypeIsFIRRTL() { }
  // expected-error @+1 {{fake_op' op found unhandled FIRRTL type}}
  %1 = "fake_op"() : () -> (!firrtl.uint<1>)
}

// -----

firrtl.circuit "RecursiveCheck" {
  firrtl.module @RecursiveCheck() { }
  func.func private @CheckRecursive() {
    // expected-error @+1 {{fake_op' op found unhandled FIRRTL type}}
    %1 = "fake_op"() : () -> (!firrtl.uint<1>)
  }
}

// -----

firrtl.circuit "BlockArgType" {
  firrtl.module @BlockArgType() { }
  // expected-error @+1 {{fake_op' op found unhandled FIRRTL type}}
  "fake_op"() ({
    ^bb0(%a: !firrtl.uint<1>):
      "fake_return"() : () -> ()
    }): () -> ()
}

// -----

firrtl.circuit "unprocessedAnnotations" {
 firrtl.module @bar(in %io_cpu_flush: !firrtl.uint<1>){
  }
  firrtl.module @unprocessedAnnotations(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                                        in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<2>,
                                        in %io_cpu_flush: !firrtl.uint<1>) {
    // expected-warning @+1 {{unprocessed annotation:'firrtl.transforms.RemainingAnnotation1'}}
    %1 = firrtl.wire {annotations = [{class = "firrtl.transforms.RemainingAnnotation1"}]} : !firrtl.uint<1>

    // expected-warning @+1 {{unprocessed annotation:'firrtl.transforms.RemainingAnnotation2'}}
    %2 = firrtl.node %1 {annotations = [{class = "firrtl.transforms.RemainingAnnotation2"}]} : !firrtl.uint<1>

    // expected-warning @+1 {{unprocessed annotation:'firrtl.transforms.RemainingAnnotation3'}}
    %3 = firrtl.reg %clock {annotations = [{class = "firrtl.transforms.RemainingAnnotation3"}]} : !firrtl.clock, !firrtl.uint<1>

    // expected-warning @+1 {{unprocessed annotation:'firrtl.transforms.RemainingAnnotation4'}}
    %4 = firrtl.regreset %clock, %reset, %1 {annotations = [{class = "firrtl.transforms.RemainingAnnotation4"}]} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>

    // expected-warning @+1 {{unprocessed annotation:'firrtl.transforms.RemainingAnnotation5'}}
    %_M_read = firrtl.mem Undefined {depth = 12 : i64, name = "_M", portNames = ["read"], readLatency = 0 : i32, writeLatency = 1 : i32, annotations =
      [{class = "firrtl.transforms.RemainingAnnotation5"}]} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>
    %_M_read.clk = firrtl.subfield %_M_read[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>
    %_M_read.en = firrtl.subfield %_M_read[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>
    %_M_read.addr = firrtl.subfield %_M_read[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>
    %_M_read.data = firrtl.subfield %_M_read[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>
    %c0_i1 = firrtl.constant 0 : !firrtl.uint<1>
    %c0_i4 = firrtl.constant 0 : !firrtl.uint<4>
    firrtl.matchingconnect %_M_read.clk, %clock : !firrtl.clock
    firrtl.matchingconnect %_M_read.en, %c0_i1 : !firrtl.uint<1>
    firrtl.matchingconnect %_M_read.addr, %c0_i4 : !firrtl.uint<4>

    // expected-warning @+1 {{unprocessed annotation:'firrtl.transforms.RemainingAnnotation6'}}
    %5 = firrtl.instance fetch {annotations = [{class = "firrtl.transforms.RemainingAnnotation6"}]} @bar(in io_cpu_flush: !firrtl.uint<1>)
    firrtl.connect %5, %io_cpu_flush : !firrtl.uint<1>, !firrtl.uint<1>

    %6 = firrtl.node %1 {annotations = [{class = "firrtl.transforms.RemainingAnnotation3"}]} : !firrtl.uint<1>
  }
}

// -----

// expected-warning @+1 {{unprocessed annotation:'circuitOpAnnotation'}}
firrtl.circuit "moduleAnno" attributes {annotations = [{class = "circuitOpAnnotation"}]} {
  // expected-warning @+1 {{unprocessed annotation:'a'}}
  firrtl.module @moduleAnno(in %io_cpu_flush: !firrtl.uint<1>) attributes
    {portAnnotations = [[{class="a"}]]} {  }
  // expected-warning @+1 {{unprocessed annotation:'b'}}
  firrtl.extmodule @extModPorts(in io_cpu_flush: !firrtl.uint<1>) attributes {portAnnotations = [[{class="b"}]]}
  // expected-warning @+1 {{unprocessed annotation:'c'}}
  firrtl.extmodule @extMod(in io_cpu_flush: !firrtl.uint<1>)
    attributes { annotations = [{class = "c"}] }
  // expected-warning @+1 {{unprocessed annotation:'d'}}
  firrtl.module @foo(in %io_cpu_flush: !firrtl.uint<1>)
    attributes { annotations = [{class = "d"}] } {}
  firrtl.module @foo2(in %io_cpu_flush: !firrtl.uint<1>)
    attributes { annotations = [{class = "b"}] } {}
  firrtl.extmodule @extModPorts2(in io_cpu_flush: !firrtl.uint<1>) attributes {portAnnotations = [[{class="c"}]]}
}

// -----

// The following annotations should be ignored and not trigger a warning
// when lowering to HW.
firrtl.circuit "Foo" attributes {annotations = [
    {class = "sifive.enterprise.firrtl.MetadataDirAnnotation", dirname = "metadata"},
    {class = "sifive.enterprise.firrtl.TestBenchDirAnnotation", dirname = "tb"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation", directory = "gct-dir", filename = "gct-dir/bindings.sv"}
  ]} {
    firrtl.module @Foo() attributes {annotations = [
        {class = "firrtl.transforms.NoDedupAnnotation"},
        {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"},
        {class = "firrtl.transforms.BlackBox", circt.nonlocal = @nla_1}
    ]} {}
    // Non-local annotations should not produce errors either.
    hw.hierpath private  @nla_1 [@Bar::@s1, @Foo]
    firrtl.module @Bar() {
      firrtl.instance foo sym @s1 {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}]} @Foo()
    }
}

// -----

firrtl.circuit "SymArgZero" {
  // expected-error @+1 {{zero width port "foo" is referenced by name [#hw<innerSym@symfoo>] (e.g. in an XMR) but must be removed}}
  firrtl.module @SymArgZero(in %foo :!firrtl.uint<0> sym @symfoo) {
  }
}

// -----

firrtl.circuit "SymWireZero" {
  firrtl.module @SymWireZero() {
    // expected-error @+2 {{couldn't handle this operation}}
    // expected-error @+1 {{zero width wire is referenced by name [#hw<innerSym@x>] (e.g. in an XMR) but must be removed}}
    %w = firrtl.wire sym [<@x,0,public>] : !firrtl.uint<0>
  }
}

// -----

firrtl.circuit "SymNodeZero" {
  firrtl.module @SymNodeZero(in %foo :!firrtl.uint<0>) {
    // expected-error @+2 {{couldn't handle this operation}}
    // expected-error @+1 {{zero width node is referenced by name [#hw<innerSym@x>] (e.g. in an XMR) but must be removed}}
    %w = firrtl.node sym [<@x,0,public>] %foo : !firrtl.uint<0>
  }
}

// -----

firrtl.circuit "DTArgZero" {
  // expected-warning @below {{zero width port "foo" has dontTouch annotation, removing anyway}}
  firrtl.module @DTArgZero(in %foo :!firrtl.uint<0> [{class = "firrtl.transforms.DontTouchAnnotation"}]) {
  }
}

// -----

firrtl.circuit "ArgWithFieldSym" {
  // expected-error @below {{cannot lower aggregate port "foo" with field sensitive symbols, HW dialect does not support per field symbols yet}}
  firrtl.module @ArgWithFieldSym(in %foo :!firrtl.vector<uint<1>,2> sym [<@x,1,public>]) {
  }
}

// -----

firrtl.circuit "ConnectDestSubfield" {
  firrtl.module @ConnectDestSubfield(in %clock: !firrtl.clock, in %value: !firrtl.uint<1>) {
    %0 = firrtl.reg %clock : !firrtl.clock, !firrtl.bundle<a: uint<1>>
    // expected-error @below {{'hw.struct_extract' op used as connect destination}}
    %1 = firrtl.subfield %0[a] : !firrtl.bundle<a: uint<1>>
    // expected-error @below {{'firrtl.matchingconnect' op LowerToHW couldn't handle this operation}}
    firrtl.matchingconnect %1, %value : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "ConnectDestSubindex" {
  firrtl.module @ConnectDestSubindex(in %clock: !firrtl.clock, in %value: !firrtl.uint<1>) {
    %0 = firrtl.reg %clock : !firrtl.clock, !firrtl.vector<uint<1>, 1>
    // expected-error @below {{'hw.array_get' op used as connect destination}}
    %1 = firrtl.subindex %0[0] : !firrtl.vector<uint<1>, 1>
    // expected-error @below {{'firrtl.matchingconnect' op LowerToHW couldn't handle this operation}}
    firrtl.matchingconnect %1, %value : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "ConnectDestSubaccess" {
  firrtl.module @ConnectDestSubaccess(in %clock: !firrtl.clock, in %index: !firrtl.uint<1>, in %value: !firrtl.uint<1>) {
    %0 = firrtl.reg %clock : !firrtl.clock, !firrtl.vector<uint<1>, 1>
    // expected-error @below {{'hw.array_get' op used as connect destination}}
    %1 = firrtl.subaccess %0[%index] : !firrtl.vector<uint<1>, 1>, !firrtl.uint<1>
    // expected-error @below {{'firrtl.matchingconnect' op LowerToHW couldn't handle this operation}}
    firrtl.matchingconnect %1, %value : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "UndrivenInputPort" {
  firrtl.extmodule private @Blackbox(in inst: !firrtl.uint<1>)

  firrtl.module @UndrivenInputPort() {
    // expected-error @below {{sink in combinational loop}}
    %0 = firrtl.instance blackbox @Blackbox(in inst : !firrtl.uint<1>)
    // expected-note @below {{through driver here}}
    %1 = firrtl.instance blackbox @Blackbox(in inst : !firrtl.uint<1>)
    firrtl.matchingconnect %0, %1 : !firrtl.uint<1>
    firrtl.matchingconnect %1, %0 : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "InputSelfDriver" {
  firrtl.module private @InputPorts(in %in : !firrtl.uint<1>) { }
  firrtl.module @InputSelfDriver(in %in : !firrtl.uint<1>) {
    // expected-error @below {{sink does not have a driver}}
    %ip2_in = firrtl.instance ip2 @InputPorts(in in : !firrtl.uint<1>)
    firrtl.connect %ip2_in, %ip2_in : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "IfElseFatalOnAssume" {
  firrtl.module @IfElseFatalOnAssume (in %clock: !firrtl.clock, in %pred: !firrtl.uint<1>, in %en: !firrtl.uint<1>) {
    // expected-error @+2 {{ifElseFatal format cannot be used for non-assertions}}
    // expected-error @below {{'firrtl.assume' op LowerToHW couldn't handle this operation}}
    firrtl.assume %clock, %en, %pred, "foo" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {isConcurrent = true, format = "ifElseFatal"}
  }
}
