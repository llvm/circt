// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-grand-central)' -split-input-file %s | FileCheck %s

firrtl.circuit "InterfaceGroundType" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        description = "description of foo",
        name = "foo",
        id = 1 : i64},
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        description = "multi\nline\ndescription\nof\nbar",
        name = "bar",
        id = 2 : i64}],
     id = 0 : i64,
     name = "View"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]} {
  firrtl.module @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Foo",
       id = 0 : i64,
       name = "View",
       type = "companion"}]} {}
  firrtl.module @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 0 : i64,
       name = "view",
       type = "parent"}
    ]} {
    %a = firrtl.wire {annotations = [
      {a},
      {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
       id = 1 : i64}]} : !firrtl.uint<2>
    %b = firrtl.wire {annotations = [
      {a},
      {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
       id = 2 : i64}]} : !firrtl.uint<4>
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @InterfaceGroundType() {
    firrtl.instance dut @DUT()
  }
}

// AugmentedBundleType is removed, ExtractGrandCentral remains.
// CHECK-LABEL: firrtl.circuit "InterfaceGroundType" {{.+}} {annotations =
// CHECK-SAME: class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation"
// CHECK-NOT: class = "sifive.enterprise.grandcentral.AugmentedBundleType"
// CHECK-SAME: {

// CHECK: firrtl.module @View_companion
// CHECK-SAME: output_file = #hw.output_file<"gct-dir/View_companion.sv"
// CHECK-NEXT: firrtl.instance View_mapping @View_mapping

// All Grand Central annotations are removed from the wires.
// CHECK: firrtl.module @DUT
// CHECK: %a = firrtl.wire
// CHECK-SAME: annotations = [{a}]
// CHECK: %b = firrtl.wire
// CHECK-SAME: annotations = [{a}]

// CHECK: firrtl.module @View_mapping
// CHECK-SAME: output_file = #hw.output_file<"gct-dir/View_mapping.sv"
// CHECK-NEXT: sv.verbatim "assign View.foo = dut.a;"
// CHECK-NEXT: sv.verbatim "assign View.bar = dut.b;"

// CHECK: sv.interface {
// CHECK-SAME: output_file = #hw.output_file<"gct-dir/Foo.sv"
// CHECK-SAME: @Foo
// CHECK-NEXT: sv.verbatim "// description of foo"
// CHECK-NEXT: sv.interface.signal @foo : i2
// CHECK-NEXT: sv.verbatim "// multi\0A// line\0A// description\0A// of\0A// bar"
// CHECK-NEXT: sv.interface.signal @bar : i4

// -----

firrtl.circuit "InterfaceVectorType" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedVectorType",
        description = "description of foo",
        elements = [
          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
           id = 1 : i64,
           name = "foo"},
          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
           id = 2 : i64,
           name = "foo"}],
        name = "foo"}],
      id = 0 : i64,
      name = "View"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]} {
  firrtl.module @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Foo",
       id = 0 : i64,
       name = "View",
       type = "companion"}]} {}
  firrtl.module @DUT(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 0 : i64,
       name = "view",
       type = "parent"}
    ]} {
    %a_0 = firrtl.reg %clock {
      annotations = [
        {a},
        {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
         id = 1 : i64}]} : !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %a_1 = firrtl.regreset %clock, %reset, %c0_ui1 {
      annotations = [
        {a},
        {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
         id = 2 : i64}]} : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @InterfaceVectorType() {
    %dut_clock, %dut_reset = firrtl.instance dut @DUT(in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
  }
}

// AugmentedBundleType is removed, ExtractGrandCentral remains.
// CHECK-LABEL: firrtl.circuit "InterfaceVectorType" {{.+}} {annotations =
// CHECK-SAME: class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation"
// CHECK-NOT: class = "sifive.enterprise.grandcentral.AugmentedBundleType"
// CHECK-SAME: {

// CHECK: firrtl.module @View_companion
// CHECK-SAME: output_file = #hw.output_file<"gct-dir/View_companion.sv"
// CHECK-NEXT: firrtl.instance View_mapping @View_mapping

// All Grand Central annotations are removed from the registers.
// CHECK: firrtl.module @DUT
// CHECK: %a_0 = firrtl.reg
// CHECK-SAME: annotations = [{a}]
// CHECK: %a_1 = firrtl.regreset
// CHECK-SAME: annotations = [{a}]

// CHECK: sv.interface {
// CHECK-SAME: output_file = #hw.output_file<"gct-dir/Foo.sv"
// CHECK-SAME: @Foo
// CHECK-NEXT: sv.verbatim "// description of foo"
// CHECK-NEXT: sv.interface.signal @foo : !hw.uarray<2xi1>

// -----

firrtl.circuit "InterfaceBundleType" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
        defName = "Bar",
        description = "description of Bar",
        elements = [
          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
           id = 1 : i64,
           name = "b"},
          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
           id = 2 : i64,
           name = "a"}],
        name = "bar"}],
     id = 0 : i64,
     name = "View"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]}  {
  firrtl.module @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Foo",
       id = 0 : i64,
       name = "View",
       type = "companion"}]} {}
  firrtl.module @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 0 : i64,
       name = "view",
       type = "parent"}
    ]} {
    %x = firrtl.wire {
      annotations = [
        {a},
        {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
         id = 2 : i64}]} : !firrtl.uint<1>
    %y = firrtl.wire {
      annotations = [
        {a},
        {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
         id = 1 : i64}]} : !firrtl.uint<2>
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @InterfaceBundleType() {
    firrtl.instance dut @DUT()
  }
}

// AugmentedBundleType is removed, ExtractGrandCentral remains.
// CHECK-LABEL: firrtl.circuit "InterfaceBundleType"
// CHECK-SAME: class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation"
// CHECK-NOT: class = "sifive.enterprise.grandcentral.AugmentedBundleType"
// CHECK-SAME: {

// All Grand Central annotations are removed from the wires.
// CHECK-LABEL: firrtl.module @DUT
// CHECK: %x = firrtl.wire
// CHECK-SAME: annotations = [{a}]
// CHECK: %y = firrtl.wire
// CHECK-SAME: annotations = [{a}]

// CHECK: sv.interface {
// CHECK-SAME: output_file = #hw.output_file<"gct-dir/Foo.sv"
// CHECK-SAME: @Foo
// CHECK-NEXT: sv.verbatim "// description of Bar"
// CHECK-NEXT: Bar bar();

// CHECK: sv.interface {
// CHECK-SAME: output_file = #hw.output_file<"gct-dir/Bar.sv"
// CHECK-SAME: @Bar
// CHECK-NEXT: sv.interface.signal @b : i2
// CHECK-NEXT: sv.interface.signal @a : i1

// -----

firrtl.circuit "InterfaceVecOfBundleType" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedVectorType",
        elements = [
          {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
           defName = "Bar",
           elements = [
             {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              id = 1 : i64,
              name = "b"}],
           name = "bar"}],
        name = "bar"}],
     id = 0 : i64,
     name = "View"}]}  {
  firrtl.module @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Foo",
       id = 0 : i64,
       name = "View",
       type = "companion"}]} {}
  firrtl.module @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 0 : i64,
       name = "view",
       type = "parent"}
    ]} {
    %x = firrtl.wire {
      annotations = [
        {a},
        {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
         id = 1 : i64}]} : !firrtl.uint<2>
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @InterfaceVecOfBundleType() {
    firrtl.instance dut @DUT()
  }
}

// CHECK-LABEL: firrtl.circuit "InterfaceVecOfBundleType"

// CHECK: sv.interface @Foo
// CHECK-NEXT: sv.verbatim "Bar bar[1]();"

// CHECK: sv.interface @Bar

// -----

firrtl.circuit "VecOfVec" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedVectorType",
        description = "description of foo",
        elements = [
          {class = "sifive.enterprise.grandcentral.AugmentedVectorType",
           description = "description of foo",
           elements = [
             {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              id = 1 : i64,
              name = "foo"},
             {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              id = 2 : i64,
              name = "foo"}],
           name = "foo"}],
        name = "foo"}],
      id = 0 : i64,
      name = "View"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]} {
  firrtl.module @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Foo",
       id = 0 : i64,
       name = "View",
       type = "companion"}]} {}
  firrtl.module @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 0 : i64,
       name = "view",
       type = "parent"}
    ]} {
    %a = firrtl.wire {annotations = [
      {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
       id = 1 : i64},
      {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
       id = 2 : i64}]} : !firrtl.uint<3>
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @VecOfVec() {
    firrtl.instance dut @DUT()
  }
}

// CHECK-LABEL: firrtl.circuit "VecOfVec"

// CHECK:      firrtl.module @View_mapping
// CHECK-NEXT:    assign View.foo[0][0]
// CHECK-NEXT:    assign View.foo[0][1]

// CHECK:      sv.interface {{.+}} @Foo
// CHECK:        sv.interface.signal @foo : !hw.uarray<1xuarray<2xi3>>

// -----

firrtl.circuit "InterfaceNode" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        description = "some expression",
        id = 1 : i64,
        name = "foo"}],
     id = 0 : i64,
     name = "View"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]} {
  firrtl.module @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Foo",
       id = 0 : i64,
       name = "View",
       type = "companion"}]} {}
  firrtl.module @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 0 : i64,
       name = "view",
       type = "parent"}
    ]} {
    %a = firrtl.wire : !firrtl.uint<2>
    %notA = firrtl.not %a : (!firrtl.uint<2>) -> !firrtl.uint<2>
    %b = firrtl.node %notA {
      annotations = [
        {a},
        {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
         defName = "Foo",
         name = "foo",
         id = 1 : i64}]} : !firrtl.uint<2>
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @InterfaceNode() {
    firrtl.instance dut @DUT()
  }
}

// AugmentedBundleType is removed, ExtractGrandCentral remains.
// CHECK-LABEL: firrtl.circuit "InterfaceNode" {{.+}} {annotations =
// CHECK-SAME: class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation"
// CHECK-NOT: class = "sifive.enterprise.grandcentral.AugmentedBundleType"
// CHECK-SAME: {

// The Grand Central annotation is removed from the node.
// CHECK: firrtl.node
// CHECK-SAME: annotations = [{a}]

// CHECK: sv.interface {
// CHECK-SAME: output_file = #hw.output_file<"gct-dir/Foo.sv"
// CHECK-SAME: @Foo
// CHECK-NEXT: sv.verbatim "// some expression"
// CHECK-NEXT: sv.interface.signal @foo : i2

// -----

firrtl.circuit "InterfacePort" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        description = "description of foo",
        id = 1 : i64,
        name = "foo"}],
     id = 0 : i64,
     name = "View"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]} {
  firrtl.module @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Foo",
       id = 0 : i64,
       name = "View",
       type = "companion"}]} {}
  firrtl.module @DUT(in %a : !firrtl.uint<4>) attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 0 : i64,
       name = "view",
       type = "parent"}
    ],
    portAnnotations = [[
      {a},
      {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
       id = 1 : i64}]] } {
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @InterfacePort() {
    %dut_a = firrtl.instance dut @DUT(in a : !firrtl.uint<4>)
  }
}

// AugmentedBundleType is removed, ExtractGrandCentral remains.
// CHECK-LABEL: firrtl.circuit "InterfacePort" {{.+}} {annotations =
// CHECK-SAME: class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation"
// CHECK-NOT: class = "sifive.enterprise.grandcentral.AugmentedBundleType"
// CHECK-SAME: {

// The Grand Central annotations are removed.
// CHECK: firrtl.module @DUT
// CHECK-SAME: %a: !firrtl.uint<4> [{a}]

// CHECK: sv.interface {
// CHECK-SAME: output_file = #hw.output_file<"gct-dir/Foo.sv"
// CHECK-SAME: @Foo
// CHECK-NEXT: sv.verbatim "// description of foo"
// CHECK-NEXT: sv.interface.signal @foo : i4

// -----

firrtl.circuit "UnsupportedTypes" attributes {
  annotations = [
    {a},
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedStringType",
        name = "string"},
       {class = "sifive.enterprise.grandcentral.AugmentedBooleanType",
        name = "boolean"},
       {class = "sifive.enterprise.grandcentral.AugmentedIntegerType",
        name = "integer"},
       {class = "sifive.enterprise.grandcentral.AugmentedDoubleType",
        name = "double"}],
     id = 0 : i64,
     name = "View"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]} {
  firrtl.module @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Foo",
       id = 0 : i64,
       name = "View",
       type = "companion"}]} {}
  firrtl.module @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 0 : i64,
       name = "view",
       type = "parent"}
    ]} {
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @UnsupportedTypes() {
    firrtl.instance dut @DUT()
  }
}

// AugmentedBundleType is removed, ExtractGrandCentral and {a} remain.
// CHECK-LABEL: firrtl.circuit "UnsupportedTypes" {{.+}} {annotations =
// CHECK-SAME: {a}
// CHECK-SAME: class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation"
// CHECK-NOT: class = "sifive.enterprise.grandcentral.AugmentedBundleType"
// CHECK-SAME: {

// CHECK: sv.interface {
// CHECK-SAME: output_file = #hw.output_file<"gct-dir/Foo.sv"
// CHECK-SAME: @Foo
// CHECK-NEXT: sv.verbatim "// <unsupported string type> string;"
// CHECK-NEXT: sv.verbatim "// <unsupported boolean type> boolean;"
// CHECK-NEXT: sv.verbatim "// <unsupported integer type> integer;"
// CHECK-NEXT: sv.verbatim "// <unsupported double type> double;"

// -----

firrtl.circuit "BindInterfaceTest"  attributes {
  annotations = [{
    class = "sifive.enterprise.grandcentral.AugmentedBundleType",
    defName = "InterfaceName",
    elements = [{
      class = "sifive.enterprise.grandcentral.AugmentedGroundType",
      id = 1 : i64,
      name = "_a"
    }],
    id = 0 : i64,
    name = "View"
  },
  {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]} {
  firrtl.module @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Foo",
       id = 0 : i64,
       name = "View",
       type = "companion"}]} {}
  firrtl.module @DUT(
    in %a: !firrtl.uint<8>, out %b: !firrtl.uint<8>) attributes {
      annotations = [{
        class = "sifive.enterprise.grandcentral.ViewAnnotation",
        defName = "InterfaceName",
        id = 0 : i64,
        name = "instanceName",
        type = "parent"
      }],
      portAnnotations = [[
        #firrtl.subAnno<fieldID = 0, {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 1 : i64}>
      ], []
      ]
    }
      {
    firrtl.connect %b, %a : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @BindInterfaceTest() {
    %dut_a, %dut_b = firrtl.instance dut @DUT(in a: !firrtl.uint<8>, out b: !firrtl.uint<8>)
  }
}

// The bind is dropped in the outer module, outside the circuit.
// CHECK: module {
// CHECK-NEXT: sv.bind.interface @[[INTERFACE_INSTANCE_SYMBOL:.+]] {output_file

// CHECK-LABEL: firrtl.circuit "BindInterfaceTest"

// Annotations are removed from the circuit.
// CHECK-NOT: annotations
// CHECK-SAME: {

// Annotations are removed from the module.
// CHECK: firrtl.module @DUT
// CHECK-NOT: annotations
// CHECK-SAME: %a

// An instance of the interface was added to the module.
// CHECK: sv.interface.instance sym @[[INTERFACE_INSTANCE_SYMBOL]] {
// CHECK-SAME: doNotPrint = true

// The interface is added.
// CHECK: sv.interface {
// CHECK-SAME: output_file = #hw.output_file<"gct-dir/InterfaceName.sv"
// CHECK-SAME: @InterfaceName
// CHECK-NEXT: sv.interface.signal @_a : i8

// -----

firrtl.circuit "MultipleGroundTypeInterfaces" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        name = "foo",
        id = 1 : i64}],
     id = 0 : i64,
     name = "View1"},
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Bar",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        name = "foo",
        id = 3 : i64}],
     id = 2 : i64,
     name = "View2"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]} {
  firrtl.module @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Foo",
       id = 0 : i64,
       name = "View1",
       type = "companion"},
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Bar",
       id = 2 : i64,
       name = "View2",
       type = "companion"}]} {}
  firrtl.module @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 0 : i64,
       name = "view",
       type = "parent"},
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 2 : i64,
       name = "view",
       type = "parent"}
    ]} {
    %a = firrtl.wire {annotations = [
      {a},
      {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
       id = 1 : i64},
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
       id = 3 : i64}]} : !firrtl.uint<2>
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @MultipleGroundTypeInterfaces() {
    firrtl.instance dut @DUT()
  }
}

// CHECK: sv.interface {
// CHECK-SAME: output_file = #hw.output_file<"gct-dir/Foo.sv"
// CHECK-SAME: @Foo

// CHECK: sv.interface {
// CHECK-SAME: output_file = #hw.output_file<"gct-dir/Bar.sv"
// CHECK-SAME: @Bar

// -----

firrtl.circuit "PrefixInterfacesAnnotation"
  attributes {annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
        defName = "Bar",
        elements = [],
        name = "bar"}],
     id = 0 : i64,
     name = "MyView"},
    {class = "sifive.enterprise.grandcentral.PrefixInterfacesAnnotation",
     prefix = "PREFIX_"}]}  {
  firrtl.module @MyView_companion()
    attributes {annotations = [{
      class = "sifive.enterprise.grandcentral.ViewAnnotation",
      id = 0 : i64,
      name = "MyView",
      type = "companion"}]} {}
  firrtl.module @DUT()
    attributes {annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 0 : i64,
       name = "MyView",
       type = "parent"}]} {
    firrtl.instance MyView_companion  @MyView_companion()
  }
  firrtl.module @PrefixInterfacesAnnotation() {
    firrtl.instance dut @DUT()
  }
}

// CHECK-LABEL: firrtl.circuit "PrefixInterfacesAnnotation"
// The PrefixInterfacesAnnotation was removed from the circuit.
// CHECK-NOT:     sifive.enterprise.grandcentral.PrefixInterfacesAnnotation

// Interface "Foo" is prefixed.
// CHECK:       sv.interface @PREFIX_Foo {
// Interface "Bar" is prefixed, but not its name.
// CHECK-NEXT:    PREFIX_Bar bar()

// Interface "Bar" is prefixed.
// CHECK:       sv.interface @PREFIX_Bar

// -----


firrtl.circuit "NestedInterfaceVectorTypes" attributes {annotations = [
  {
    class = "sifive.enterprise.grandcentral.AugmentedBundleType",
    id = 0,
    name = "View",
    defName = "Foo",
    elements = [{
      class = "sifive.enterprise.grandcentral.AugmentedVectorType",
      name = "bar",
      description = "description of bar",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
          name = "baz",
          elements = [
            {class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 1, name = "baz"},
            {class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 2, name = "baz"},
            {class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 3, name = "baz"}
          ]
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
          name = "baz",
          elements = [
            {class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 4, name = "baz"},
            {class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 5, name = "baz"},
            {class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 6, name = "baz"}
          ]
        }
      ]
    }]
  },
  {
    class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
    directory = "gct-dir",
    filename = "gct-dir/bindings.sv"
  }
]} {

  firrtl.module @View_companion() attributes {annotations = [
    {class = "sifive.enterprise.grandcentral.ViewAnnotation", defName = "Foo", id = 0, name = "View", type = "companion"}
  ]} {}

  firrtl.module @DUT() attributes {annotations = [
    {class = "sifive.enterprise.grandcentral.ViewAnnotation", id = 0, name = "view", type = "parent"}
  ]} {
    %a0 = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 1}]} : !firrtl.uint<1>
    %a1 = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 2}]} : !firrtl.uint<1>
    %a2 = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 3}]} : !firrtl.uint<1>
    %b0 = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 4}]} : !firrtl.uint<1>
    %b1 = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 5}]} : !firrtl.uint<1>
    %b2 = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 6}]} : !firrtl.uint<1>
    firrtl.instance View_companion @View_companion()
  }

  firrtl.module @NestedInterfaceVectorTypes() {
    firrtl.instance dut @DUT()
  }
}

// CHECK-LABEL: firrtl.circuit "NestedInterfaceVectorTypes"
// CHECK:         firrtl.module @View_mapping
// CHECK-NEXT:      sv.verbatim "assign View.bar[0][0] = dut.a0;"
// CHECK-NEXT:      sv.verbatim "assign View.bar[0][1] = dut.a1;"
// CHECK-NEXT:      sv.verbatim "assign View.bar[0][2] = dut.a2;"
// CHECK-NEXT:      sv.verbatim "assign View.bar[1][0] = dut.b0;"
// CHECK-NEXT:      sv.verbatim "assign View.bar[1][1] = dut.b1;"
// CHECK-NEXT:      sv.verbatim "assign View.bar[1][2] = dut.b2;"
// CHECK:         sv.interface {
// CHECK-SAME:      @Foo
// CHECK-NEXT:      sv.verbatim "// description of bar"
// CHECK-NEXT:      sv.interface.signal @bar : !hw.uarray<2xuarray<3xi1>>

// -----

firrtl.circuit "VerbatimTypesInVector" attributes {annotations = [
  {
    class = "sifive.enterprise.grandcentral.AugmentedBundleType",
    id = 0,
    name = "View",
    defName = "Foo",
    elements = [{
      class = "sifive.enterprise.grandcentral.AugmentedVectorType",
      name = "bar",
      description = "description of bar",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
          name = "baz",
          description = "description of baz",
          elements = [
            {class = "sifive.enterprise.grandcentral.AugmentedStringType", name = "baz"},
            {class = "sifive.enterprise.grandcentral.AugmentedStringType", name = "baz"},
            {class = "sifive.enterprise.grandcentral.AugmentedStringType", name = "baz"}
          ]
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
          name = "baz",
          description = "description of baz",
          elements = [
            {class = "sifive.enterprise.grandcentral.AugmentedStringType", name = "baz"},
            {class = "sifive.enterprise.grandcentral.AugmentedStringType", name = "baz"},
            {class = "sifive.enterprise.grandcentral.AugmentedStringType", name = "baz"}
          ]
        }
      ]
    }]
  },
  {
    class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
    directory = "gct-dir",
    filename = "gct-dir/bindings.sv"
  }
]} {

  firrtl.module @View_companion() attributes {annotations = [
    {class = "sifive.enterprise.grandcentral.ViewAnnotation", defName = "Foo", id = 0, name = "View", type = "companion"}
  ]} {}

  firrtl.module @DUT() attributes {annotations = [
    {class = "sifive.enterprise.grandcentral.ViewAnnotation", id = 0, name = "view", type = "parent"}
  ]} {
    firrtl.instance View_companion @View_companion()
  }

  firrtl.module @VerbatimTypesInVector() {
    firrtl.instance dut @DUT()
  }
}

// CHECK-LABEL: firrtl.circuit "VerbatimTypesInVector"
// CHECK:         sv.interface {
// CHECK-SAME:      @Foo
// CHECK-NEXT:      sv.verbatim "// description of bar"
// CHECK-NEXT:      sv.verbatim "// <unsupported string type> bar[2][3];"

// -----

firrtl.circuit "YAMLOutputEmptyInterface" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [],
     id = 0 : i64,
     name = "View"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"},
    {class = "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation",
     filename = "gct-dir/gct.yaml"}]} {
  firrtl.module @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Foo",
       id = 0 : i64,
       name = "View",
       type = "companion"}]} {}
  firrtl.module @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 0 : i64,
       name = "view",
       type = "parent"}
    ]} {
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @YAMLOutputEmptyInterface() {
    firrtl.instance dut @DUT()
  }
}

// CHECK-LABEL: module {
// CHECK:        sv.verbatim
// CHECK-SAME:      - name: Foo
// CHECK-SAME:        fields: []
// CHECK-SAME:        instances: []
// CHECK-SAME:      {output_file = #hw.output_file<"gct-dir/gct.yaml"
//
// CHECK-NOT:  class = "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation"

// -----

firrtl.circuit "YAMLOutputTwoInterfaces" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [],
     id = 0 : i64,
     name = "View"},
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Bar",
     elements = [],
     id = 1 : i64,
     name = "View2"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"},
    {class = "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation",
     filename = "gct-dir/gct.yaml"}]} {
  firrtl.module @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Foo",
       id = 0 : i64,
       name = "View",
       type = "companion"},
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Bar",
       id = 1 : i64,
       name = "View2",
       type = "companion"}]} {}
  firrtl.module @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 0 : i64,
       name = "view",
       type = "parent"},
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 1 : i64,
       name = "view",
       type = "parent"}]} {
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @YAMLOutputTwoInterfaces() {
    firrtl.instance dut @DUT()
  }
}

// CHECK-LABEL: module {
// CHECK:        sv.verbatim
// CHECK-SAME:      - name: Foo
// CHECK-SAME:        fields: []
// CHECK-SAME:        instances: []
// CHECK-SAME:      - name: Bar
// CHECK-SAME:        fields: []
// CHECK-SAME:        instances: []

// -----

firrtl.circuit "YAMLOutputScalarField" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        description = "description of foo",
        name = "foo",
        id = 1 : i64}
     ],
     id = 0 : i64,
     name = "View"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"},
    {class = "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation",
     filename = "gct-dir/gct.yaml"}]} {
  firrtl.module @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Foo",
       id = 0 : i64,
       name = "View",
       type = "companion"}]} {}
  firrtl.module @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 0 : i64,
       name = "view",
       type = "parent"}
    ]} {
    %a = firrtl.wire {annotations = [
      {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
       id = 1 : i64}]} : !firrtl.uint<2>
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @YAMLOutputScalarField() {
    firrtl.instance dut @DUT()
  }
}

// CHECK-LABEL: module {
// CHECK:        sv.verbatim
// CHECK-SAME:      - name: Foo
// CHECK-SAME:        fields:
// CHECK-SAME:          - name: foo
// CHECK-SAME:            description: description of foo
// CHECK-SAME:            dimensions: [ ]
// CHECK-SAME:            width: 2
// CHECK-SAME:        instances: []
//
// Note: Built-in vector serialization works slightly differently than
// user-defined vector serialization.  This results in the verbose "[ ]" for the
// empty dimensions vector, and the terse "[]" for the empty instances vector.

// -----

firrtl.circuit "YAMLOutputVectorField" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedVectorType",
        elements = [
          {class = "sifive.enterprise.grandcentral.AugmentedVectorType",
           elements = [
             {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
             id = 1 : i64,
             name = "foo"}],
           name = "foo"
          },
          {class = "sifive.enterprise.grandcentral.AugmentedVectorType",
           elements = [
             {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
             id = 2 : i64,
             name = "foo"}],
           name = "foo"
          }],
        name = "foo"}],
     id = 0 : i64,
     name = "View"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"},
    {class = "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation",
     filename = "gct-dir/gct.yaml"}]} {
  firrtl.module @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Foo",
       id = 0 : i64,
       name = "View",
       type = "companion"}]} {}
  firrtl.module @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 0 : i64,
       name = "view",
       type = "parent"}
    ]} {
    %a = firrtl.wire {annotations = [
      {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
       id = 1 : i64},
      {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
       id = 2 : i64}]} : !firrtl.uint<8>
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @YAMLOutputVectorField() {
    firrtl.instance dut @DUT()
  }
}

// CHECK-LABEL: module {
// CHECK:        sv.verbatim
// CHECK-SAME:      - name: Foo
// CHECK-SAME:        fields:
// CHECK-SAME:          - name: foo
// CHECK-SAME:            dimensions: [ 1, 2 ]
// CHECK-SAME:            width: 8
// CHECK-SAME:        instances: []

// -----

firrtl.circuit "YAMLOutputInstance" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
        defName = "Bar",
        elements = [
          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
           id = 1 : i64,
           name = "baz"}],
        name = "bar"}],
     id = 0 : i64,
     name = "View"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"},
    {class = "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation",
     filename = "gct-dir/gct.yaml"}]} {
  firrtl.module @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Foo",
       id = 0 : i64,
       name = "View",
       type = "companion"}]} {}
  firrtl.module @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 0 : i64,
       name = "view",
       type = "parent"}
    ]} {
    %a = firrtl.wire {annotations = [
      {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
       id = 1 : i64},
      {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
       id = 2 : i64}]} : !firrtl.uint<8>
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @YAMLOutputInstance() {
    firrtl.instance dut @DUT()
  }
}

// CHECK-LABEL: module {
// CHECK:        sv.verbatim
// CHECK-SAME:      - name: Foo
// CHECK-SAME:        fields: []
// CHECK-SAME:        instances:
// CHECK-SAME:          - name: bar
// CHECK-SAME:            dimensions: [ ]
// CHECK-SAME:            interface:
// CHECK-SAME:              name: Bar
// CHECK-SAME:              fields:
// CHECK-SAME:                - name: baz
// CHECK-SAME:                  dimensions: [ ]
// CHECK-SAME:                  width: 8
// CHECK-SAME:              instances: []
