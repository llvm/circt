// RUN: circt-opt -om-freeze-paths --split-input-file --verify-diagnostics %s

hw.hierpath private @nla [@Top::@sym]
hw.module @Top() {
  // expected-note @below {{component here}}
  %wire = hw.wire %wire sym @sym : i8
  // expected-error @below {{component does not have verilog name}}
  %path = om.path reference @nla
  hw.output
}

// -----

hw.hierpath private @nla [@Child]
hw.module private @Child() {}
hw.module @Top() {
  // expected-error @below {{unable to uniquely resolve target due to multiple instantiation}}
  %path = om.path reference @nla
  // expected-note @below {{instance here}}
  hw.instance "child0" @Child() -> ()
  // expected-note @below {{instance here}}
  hw.instance "child1" @Child() -> ()
  hw.output
}
