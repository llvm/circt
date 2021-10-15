// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-grand-central)' -split-input-file -verify-diagnostics %s

// expected-error @+1 {{more than one 'ExtractGrandCentralAnnotation' was found, but exactly one must be provided}}
firrtl.circuit "MoreThanOneExtractGrandCentralAnnotation" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {name = "foo",
        tpe = "sifive.enterprise.grandcentral.AugmentedGroundType"}]},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}] } {
  firrtl.module @MoreThanOneExtractGrandCentralAnnotation() {}
}

// -----

firrtl.circuit "NonGroundType" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
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
    // expected-error @+1 {{'firrtl.wire' op cannot be added to interface with id '0' because it is not a ground type}}
    %a = firrtl.wire {
      annotations = [
        {a},
        {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
         id = 1 : i64}]} : !firrtl.vector<uint<2>, 1>
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @NonGroundType() {
    firrtl.instance dut @DUT()
  }
}

// -----

// expected-error @+1 {{missing 'id' in root-level BundleType}}
firrtl.circuit "NonGroundType" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]} {
  firrtl.module @NonGroundType() {}
}

// -----

firrtl.circuit "Foo" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "View",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        id = 1 : i64}],
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
  firrtl.module @Bar(in %a: !firrtl.uint<1>) {}
  firrtl.module @DUT(in %a: !firrtl.uint<1>) attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 0 : i64,
       name = "View",
       type = "parent"}]} {
    // expected-error @+1 {{'firrtl.instance' op is marked as an interface element, but this should be impossible due to how the Chisel Grand Central API works}}
    %bar_a = firrtl.instance bar @Bar(in a: !firrtl.uint<1> [
        {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
         id = 1 : i64}])
    firrtl.connect %bar_a, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @Foo() {
    %dut_a = firrtl.instance dut @DUT(in a: !firrtl.uint<1>)
  }
}

// -----

firrtl.circuit "Foo" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "View",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        id = 1 : i64,
        name = "foo"}],
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
  firrtl.module @DUT(in %a: !firrtl.uint<1>) attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 0 : i64,
       name = "View",
       type = "parent"}]} {
    // expected-error @+1 {{'firrtl.mem' op is marked as an interface element, but this does not make sense (is there a scattering bug or do you have a malformed hand-crafted MLIR circuit?)}}
    %memory_b_r = firrtl.mem Undefined {
      annotations = [
        {a},
        {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
         id = 1 : i64}],
      depth = 16 : i64,
      name = "memory_b",
      portNames = ["r"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @Foo() {
    %dut_a = firrtl.instance dut @DUT(in a: !firrtl.uint<1>)
  }
}

// -----

// expected-error @+1 {{'firrtl.circuit' op has an AugmentedGroundType with 'id == 42' that does not have a scattered leaf to connect to in the circuit}}
firrtl.circuit "Foo" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Bar",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        id = 42 : i64,
        name = "baz"}],
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
       name = "View",
       type = "parent"}]} {
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @Foo() {
    firrtl.instance dut @DUT()
  }
}
