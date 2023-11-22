// RUN: circt-opt -om-freeze-paths --split-input-file --verify-diagnostics %s

hw.hierpath private @nla [@Top::@sym]
hw.module @Top() {
  // expected-note @below {{component here}}
  %wire = hw.wire %wire sym @sym : i8
  hw.output
}
om.class @OM(%basepath: !om.basepath) {
  // expected-error @below {{component does not have verilog name}}
  %path = om.path_create reference %basepath @nla
}

// -----

hw.hierpath private @nla [@Top::@sym]
hw.module @Top() {
  %wire = hw.wire %wire sym @sym {hw.verilogName = "wire"} : i8
  hw.output
}
om.class @OM(%basepath: !om.basepath) {
  // expected-error @below {{basepath must target an instance}}
  %path = om.basepath_create %basepath @nla
}
