// RUN: circt-opt --firrtl-probes-to-signals --verify-diagnostics --split-input-file %s

// CHECK-LABEL: "ProbeAndRWProbe"
firrtl.circuit "ProbeAndRWProbe" {
  // expected-error @below {{rwprobe not supported, cannot convert type '!firrtl.rwprobe<uint<2>>'}}
  firrtl.extmodule private @Probes(out ro : !firrtl.probe<uint<1>>, out rw : !firrtl.rwprobe<uint<2>>)
  firrtl.module @ProbeAndRWProbe() {}
}

// -----

firrtl.circuit "InternalPath" {
  // expected-error @below {{cannot convert module with internal path}}
  firrtl.extmodule @InternalPath(
      out x: !firrtl.probe<bundle<a: uint<5>, b: uint<3>>>
    ) attributes { internalPaths = [#firrtl.internalpath<"a.b.c">] }
}


// -----

// Sending probe out from under a when is not possible without encompassing ExpandWhens.
// Detect and diagnose, and in practice use ExpandWhens first to ensure success.
firrtl.circuit "RefProducer" {
  // expected-note @below {{destination here}}
  firrtl.module @RefProducer(in %a: !firrtl.uint<4>, in %en: !firrtl.uint<1>, in %clk: !firrtl.clock, out %thereg: !firrtl.probe<uint>) attributes {convention = #firrtl<convention scalarized>} {
    firrtl.when %en : !firrtl.uint<1> {
      %myreg = firrtl.reg interesting_name %clk : !firrtl.clock, !firrtl.uint
      firrtl.connect %myreg, %a : !firrtl.uint, !firrtl.uint<4>
      // expected-note @below {{source here}}
      %0 = firrtl.ref.send %myreg : !firrtl.uint
      // expected-error @below {{unable to convert to equivalent connect}}
      firrtl.ref.define %thereg, %0 : !firrtl.probe<uint>
    }
  }
}

// -----

firrtl.circuit "Forceable" {
   firrtl.module @Forceable() {
     // expected-error @below {{forceable declaration not supported}}
     %w, %w_f = firrtl.wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
   }
}

// -----

firrtl.circuit "RWProbeOp" {
  firrtl.module @RWProbeOp() {
    %w = firrtl.wire sym @sym : !firrtl.uint<2>
    // expected-error @below {{rwprobe not supported}}
    %rwprobe = firrtl.ref.rwprobe <@RWProbeOp::@sym> : !firrtl.rwprobe<uint<2>>
  }
}

// -----

firrtl.circuit "ExtOpenAgg" {
  firrtl.extmodule @ExtOpenAgg(
      // expected-error @below {{open aggregates not supported, cannot convert type}}
      out out: !firrtl.openbundle<a: bundle<data: uint<1>>, b: openvector<openbundle<x: uint<2>, y: probe<vector<uint<2>, 3>>>, 2>>)
}

// -----

firrtl.circuit "LayerProbe" {
  firrtl.layer @Layer bind {}
  // expected-error @below {{layer-colored probes not supported, cannot convert type}}
  firrtl.module @LayerProbe(out %p: !firrtl.probe<uint<1>, @Layer>) {
    firrtl.layerblock @Layer {
      %w = firrtl.wire : !firrtl.uint<1>
      %w_p = firrtl.ref.send %w : !firrtl.uint<1>
      %w_p_l = firrtl.ref.cast %w_p : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>, @Layer>
      firrtl.ref.define %p, %w_p_l : !firrtl.probe<uint<1>, @Layer>
    }
  }
}
