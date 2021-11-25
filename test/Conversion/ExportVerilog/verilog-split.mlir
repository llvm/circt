// RUN: circt-opt --export-split-verilog --verify-diagnostics %s

// The following used to trigger emission of `sv.bind` within a parallel block,
// triggering an assertion.
hw.module @Foo() {}
hw.module @Top() {
  hw.instance "BindEmissionInstance" sym @instA @Foo() -> () {doNotPrint = true}
}
sv.bind #hw.innerNameRef<@Top::@instA> {output_file = #hw.output_file<"foo.sv", excludeFromFileList>}
