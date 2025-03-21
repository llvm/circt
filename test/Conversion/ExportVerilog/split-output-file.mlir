// RUN: circt-opt %s --export-split-verilog='dir-name=%t' | FileCheck %s

// COM: Check explicit output_file.
// CHECK: hw.module @foo
// CHECK-SAME: output_file = #hw.output_file<"alternate.sv", includeReplicatedOps>
#fooFile = #hw.output_file<"alternate.sv", includeReplicatedOps>
hw.module @foo(in %a: i1, out b: i1) attributes {output_file = #fooFile} {
  hw.output %a : i1
}

// COM: Check back-annotated output_file.
// CHECK: hw.module @bar
// CHECK-SAME: output_file = #hw.output_file<"bar.sv", includeReplicatedOps>
hw.module @bar(in %a: i1, out b: i1) {
  hw.output %a : i1
}
