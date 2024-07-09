// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-annotations{should-enable-debug-info=true}))' --split-input-file %s | FileCheck --check-prefix=CHECK_DBG %s
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-annotations{should-enable-debug-info=false}))' --split-input-file %s | FileCheck --check-prefix=CHECK_NO_DBG %s


// Annotating a module
// Test tywaves annotations
// CHECK-LABEL: firrtl.circuit "TywavesTop"
firrtl.circuit "TywavesTop" attributes {
  rawAnnotations = [
    {
      // Annotation of a module
      class = "chisel3.tywavesinternal.TywavesAnnotation",
      target = "~TywavesTop|DUT",
      typeName = "TywavesDut",
      params=[
        { "name"="size", "typeName"="Int", "value"="10" },
        { "name"="n", "typeName"="Int" }
      ]
    }]
  } {
    // CHECK_DBG:  firrtl.module private @DUT() attributes
    // CHECK_NO_DBG:  firrtl.module private @DUT() {
    // CHECK_DBG:        annotations = [{
    // CHECK_DBG:           class = "chisel3.tywavesinternal.TywavesAnnotation"
    // CHECK_DBG:           params = [{name = "size", typeName = "Int", value = "10"}, {name = "n", typeName = "Int"}
    // CHECK_DBG:           target = "~TywavesTop|DUT"
    // CHECK_DBG:           typeName = "TywavesDut"
  firrtl.module private @DUT() {}
    // CHECK-LABEL:  firrtl.module @TywavesTop
    // CHECK-NEXT:   firrtl.instance dut @DUT
    firrtl.module @TywavesTop() {
    firrtl.instance dut @DUT()
  }
}

// CHECK_DBG: firrtl.circuit "Foo"
firrtl.circuit "Foo" attributes {rawAnnotations = [
    {
        class = "chisel3.tywavesinternal.TywavesAnnotation",
        target = "~Foo|Foo>bar.a",
        typeName = "UInt<1>"
    },
    {
        class = "chisel3.tywavesinternal.TywavesAnnotation",
        target = "~Foo|Foo>bar.b.baz",
        typeName = "UInt<1>"
    },
    {
        class = "chisel3.tywavesinternal.TywavesAnnotation",
        target = "~Foo|Foo/bar:Bar>b.qux",
        typeName = "UInt<1>"
    },
    {
        class = "chisel3.tywavesinternal.TywavesAnnotation",
        target = "~Foo|Foo/bar:Bar>d.qux",
        typeName = "UInt<1>"
    },
    {
        class = "chisel3.tywavesinternal.TywavesAnnotation",
        target = "~Foo|Foo>bar.c",
        typeName = "Bool"
    }
]} {
  // CHECK-LABEL:    firrtl.module @Bar
  // CHECK_NO_DBG: in %a: !firrtl.uint<1>, out %b: !firrtl.bundle<baz: uint<1>, qux: uint<1>>, out %c: !firrtl.uint<1>
  // CHECK-SAME:   in %a
  // CHECK_DBG:     {class = "chisel3.tywavesinternal.TywavesAnnotation", target = "~Foo|Foo>bar.a", typeName = "UInt<1>"}
  // CHECK-SAME:   out %b
  // CHECK_DBG:     {circt.fieldID = 1 : i32, class = "chisel3.tywavesinternal.TywavesAnnotation", target = "~Foo|Foo>bar.b.baz", typeName = "UInt<1>"}
  // CHECK_DBG:     {circt.fieldID = 2 : i32, class = "chisel3.tywavesinternal.TywavesAnnotation", target = "~Foo|Foo/bar:Bar>b.qux", typeName = "UInt<1>"}
  // CHECK-SAME:   out %c
  // CHECK_DBG:     {class = "chisel3.tywavesinternal.TywavesAnnotation", target = "~Foo|Foo>bar.c", typeName = "Bool"}

  firrtl.module @Bar(
    in %a: !firrtl.uint<1>,
    out %b: !firrtl.bundle<baz: uint<1>, qux: uint<1>>,
    out %c: !firrtl.uint<1>
  ) {
    // CHECK-NEXT: %d = firrtl.wire
    // CHECK_DBG:   {circt.fieldID = 2 : i32, class = "chisel3.tywavesinternal.TywavesAnnotation", target = "~Foo|Foo/bar:Bar>d.qux", typeName = "UInt<1>"}
    %d = firrtl.wire : !firrtl.bundle<baz: uint<1>, qux: uint<1>>
  }
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() {
    %bar_a, %bar_b, %bar_c = firrtl.instance bar @Bar(
      in a: !firrtl.uint<1>,
      out b: !firrtl.bundle<baz: uint<1>, qux: uint<1>>,
      out c: !firrtl.uint<1>
    )
  }
}