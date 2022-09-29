// RUN:  circt-opt --sv-extract-test-code --debug-only=extract-test-code %s -verify-diagnostics
module attributes {firrtl.extract.assert =  #hw.output_file<"dir3/", excludeFromFileList, includeReplicatedOps>, firrtl.extract.assume.bindfile = #hw.output_file<"file4", excludeFromFileList>, firrtl.extract.testbench = #hw.output_file<"testbench/", excludeFromFileList, includeReplicatedOps>} {
  hw.module.extern @foo_cover(%a : i1) -> (b : i1) attributes {"firrtl.extract.cover.extra"} 
  hw.module @extract_return(%clock: i1) -> (c : i1) {
// expected-error @+1 {{Extracting op with result}}
    %b = hw.instance "bar_cover" @foo_cover(a: %clock : i1) -> (b : i1)
    hw.output %b : i1
  }
// expected-warning @+1 {{already bound, skipping}}
  hw.module @input_only(%clock: i1, %cond: i1) -> () {
    sv.always posedge %clock  {
      sv.assert %cond, immediate
      sv.assume %cond, immediate
    }
  }
  hw.module @input_only_top(%clock: i1, %cond: i1) -> (foo: i1) {
    hw.instance "input_only" @input_only(clock: %clock: i1, cond: %cond: i1) -> ()
    hw.output %cond : i1
  }
}
