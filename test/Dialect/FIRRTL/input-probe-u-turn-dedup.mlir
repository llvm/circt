// Ensure composition of hoist-passthrough + probe-dce allows dedup of modules
// w/input probes.
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-hoist-passthrough,firrtl-probe-dce,firrtl-dedup))' %s

firrtl.circuit "UTurnDedup" attributes {annotations = [{
    class = "firrtl.transforms.MustDeduplicateAnnotation",
    modules = ["~UTurnDedup|UTurn1", "~MustDedup|UTurn2"]}]
   } {
  firrtl.module private @UTurn1(in %in : !firrtl.probe<uint<5>>,
                               out %out : !firrtl.probe<uint<5>>) {
    firrtl.ref.define %out, %in : !firrtl.probe<uint<5>>
  }
  firrtl.module private @UTurn2(in %in : !firrtl.probe<uint<5>>,
                               out %out : !firrtl.probe<uint<5>>) {
    firrtl.ref.define %out, %in : !firrtl.probe<uint<5>>
  }
  firrtl.module @UTurnDedup(in %in : !firrtl.uint<5>, out %out : !firrtl.uint<5>) {
    %u1_in, %u1_out = firrtl.instance u @UTurn1(in in : !firrtl.probe<uint<5>>,
                                               out out : !firrtl.probe<uint<5>>)
    %u2_in, %u2_out = firrtl.instance u @UTurn2(in in : !firrtl.probe<uint<5>>,
                                                out out : !firrtl.probe<uint<5>>)
    %ref = firrtl.ref.send %in : !firrtl.uint<5>
    firrtl.ref.define %u1_in, %ref : !firrtl.probe<uint<5>>
    firrtl.ref.define %u2_in, %ref : !firrtl.probe<uint<5>>
    %data1 = firrtl.ref.resolve %u1_out : !firrtl.probe<uint<5>>
    %data2 = firrtl.ref.resolve %u2_out : !firrtl.probe<uint<5>>
  }
}
