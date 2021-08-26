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
        name = "bar",
        id = 2 : i64}],
     id = 0 : i64},
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
    firrtl.instance @View_companion { name = "View_companion" }
  }
  firrtl.module @InterfaceGroundType() {
    firrtl.instance @DUT {name = "dut" }
  }
}

// AugmentedBundleType is removed, ExtractGrandCentral remains.
// CHECK-LABEL: firrtl.circuit "InterfaceGroundType" {{.+}} {annotations =
// CHECK-SAME: class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation"
// CHECK-NOT: class = "sifive.enterprise.grandcentral.AugmentedBundleType"
// CHECK-SAME: {

// CHECK: firrtl.module @View_companion
// CHECK-SAME: output_file = {directory = "gct-dir"
// CHECK-SAME: name = "View_companion.sv"
// CHECK-NEXT: firrtl.instance @View_mapping {name = "View_mapping"}

// All Grand Central annotations are removed from the wires.
// CHECK: firrtl.module @DUT
// CHECK: %a = firrtl.wire
// CHECK-SAME: annotations = [{a}]
// CHECK: %b = firrtl.wire
// CHECK-SAME: annotations = [{a}]

// CHECK: firrtl.module @View_mapping
// CHECK-SAME: output_file = {directory = "gct-dir"
// CHECK-SAME: name = "View_mapping.sv"
// CHECK-NEXT: sv.verbatim "assign View.foo = dut.a;"
// CHECK-NEXT: sv.verbatim "assign View.bar = dut.b;"

// CHECK: sv.interface {
// CHECK-SAME: output_file = {directory = "gct-dir"
// CHECK-SAME: name = "Foo.sv"
// CHECK-SAME: @Foo
// CHECK-NEXT: sv.verbatim "// description of foo"
// CHECK-NEXT: sv.interface.signal @foo : i2
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
      id = 0 : i64},
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
    firrtl.instance @View_companion {name = "View_companion"}
  }
  firrtl.module @InterfaceVectorType() {
    %dut_clock, %dut_reset = firrtl.instance @DUT {name = "dut"} : !firrtl.clock, !firrtl.uint<1>
  }
}

// AugmentedBundleType is removed, ExtractGrandCentral remains.
// CHECK-LABEL: firrtl.circuit "InterfaceVectorType" {{.+}} {annotations =
// CHECK-SAME: class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation"
// CHECK-NOT: class = "sifive.enterprise.grandcentral.AugmentedBundleType"
// CHECK-SAME: {

// CHECK: firrtl.module @View_companion
// CHECK-SAME: output_file = {directory = "gct-dir"
// CHECK-SAME: name = "View_companion.sv"
// CHECK-NEXT: firrtl.instance @View_mapping {name = "View_mapping"}

// All Grand Central annotations are removed from the registers.
// CHECK: firrtl.module @DUT
// CHECK: %a_0 = firrtl.reg
// CHECK-SAME: annotations = [{a}]
// CHECK: %a_1 = firrtl.regreset
// CHECK-SAME: annotations = [{a}]

// CHECK: sv.interface {
// CHECK-SAME: output_file = {directory = "gct-dir"
// CHECK-SAME: name = "Foo.sv"
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
           name = "a"}]}],
     id = 0 : i64},
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
    firrtl.instance @View_companion {name = "View_companion"}
  }
  firrtl.module @InterfaceBundleType() {
    firrtl.instance @DUT {name = "dut"}
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
// CHECK-SAME: output_file = {directory = "gct-dir"
// CHECK-SAME: name = "Foo.sv"
// CHECK-SAME: @Foo
// CHECK-NEXT: sv.verbatim "// description of Bar"
// CHECK-NEXT: Bar Bar();

// CHECK: sv.interface {
// CHECK-SAME: output_file = {directory = "gct-dir"
// CHECK-SAME: name = "Bar.sv"
// CHECK-SAME: @Bar
// CHECK-NEXT: sv.interface.signal @b : i2
// CHECK-NEXT: sv.interface.signal @a : i1

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
     id = 0 : i64},
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
    firrtl.instance @View_companion {name = "View_companion"}
  }
  firrtl.module @InterfaceNode() {
    firrtl.instance @DUT {name = "dut"}
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
// CHECK-SAME: output_file = {directory = "gct-dir"
// CHECK-SAME: name = "Foo.sv"
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
     id = 0 : i64},
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
    firrtl.instance @View_companion {name = "View_companion"}
  }
  firrtl.module @InterfacePort() {
    %dut_a = firrtl.instance @DUT {name = "dut"} : !firrtl.uint<4>
  }
}

// AugmentedBundleType is removed, ExtractGrandCentral remains.
// CHECK-LABEL: firrtl.circuit "InterfacePort" {{.+}} {annotations =
// CHECK-SAME: class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation"
// CHECK-NOT: class = "sifive.enterprise.grandcentral.AugmentedBundleType"
// CHECK-SAME: {

// The Grand Central annotations are removed.
// CHECK: firrtl.module @DUT
// CHECK-SAME: %a: !firrtl.uint<4> {firrtl.annotations = [{a}]}

// CHECK: sv.interface {
// CHECK-SAME: output_file = {directory = "gct-dir"
// CHECK-SAME: name = "Foo.sv"
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
     id = 0 : i64},
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
    firrtl.instance @View_companion {name = "View_companion"}
  }
  firrtl.module @UnsupportedTypes() {
    firrtl.instance @DUT {name = "dut"}
  }
}

// AugmentedBundleType is removed, ExtractGrandCentral and {a} remain.
// CHECK-LABEL: firrtl.circuit "UnsupportedTypes" {{.+}} {annotations =
// CHECK-SAME: {a}
// CHECK-SAME: class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation"
// CHECK-NOT: class = "sifive.enterprise.grandcentral.AugmentedBundleType"
// CHECK-SAME: {

// CHECK: sv.interface {
// CHECK-SAME: output_file = {directory = "gct-dir"
// CHECK-SAME: name = "Foo.sv"
// CHECK-SAME: @Foo
// CHECK-NEXT: sv.verbatim "// string = <unsupported string type>;"
// CHECK-NEXT: sv.verbatim "// boolean = <unsupported boolean type>;"
// CHECK-NEXT: sv.verbatim "// integer = <unsupported integer type>;"
// CHECK-NEXT: sv.verbatim "// double = <unsupported double type>;"

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
    id = 0 : i64
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
    firrtl.instance @View_companion {name = "View_companion"}
  }
  firrtl.module @BindInterfaceTest() {
    %dut_a, %dut_b = firrtl.instance @DUT {name = "dut"} : !firrtl.uint<8>, !firrtl.uint<8>
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
// CHECK-SAME: output_file = {directory = "gct-dir"
// CHECK-SAME: name = "InterfaceName.sv"
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
     id = 0 : i64},
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Bar",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        name = "foo",
        id = 3 : i64}],
     id = 2 : i64},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]} {
  firrtl.module @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Foo",
       id = 0 : i64,
       name = "View",
       type = "companion"},
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       defName = "Bar",
       id = 2 : i64,
       name = "View",
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
    firrtl.instance @View_companion { name = "View_companion" }
  }
  firrtl.module @MultipleGroundTypeInterfaces() {
    firrtl.instance @DUT {name = "dut" }
  }
}

// CHECK: sv.interface {
// CHECK-SAME: name = "Foo.sv"
// CHECK-SAME: @Foo

// CHECK: sv.interface {
// CHECK-SAME: name = "Bar.sv"
// CHECK-SAME: @Bar
