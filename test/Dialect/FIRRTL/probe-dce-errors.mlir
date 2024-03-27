// RUN: circt-opt --firrtl-probe-dce --split-input-file --verify-diagnostics %s

// Diagnose input probes on external modules.
firrtl.circuit "ExtPort" {
  // expected-error @below {{input probe not allowed on this module kind}}
  firrtl.extmodule private @Ext(in in : !firrtl.probe<uint<1>>)
  firrtl.module @ExtPort() {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %ref = firrtl.ref.send %zero : !firrtl.uint<1>
    %in = firrtl.instance ext @Ext(in in : !firrtl.probe<uint<1>>)
    firrtl.ref.define %in, %ref : !firrtl.probe<uint<1>>
  }
}

// -----
// Diagnose use of input probe (upwards or n-turn).

firrtl.circuit "Upwards" {
  firrtl.module private @Read(
      // expected-note @below {{input probe}}
      in %in : !firrtl.probe<uint<1>>) {
    // expected-error @below {{input probes cannot be used}}
    %data = firrtl.ref.resolve %in : !firrtl.probe<uint<1>>
  }
  firrtl.module @Upwards() {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %ref = firrtl.ref.send %zero : !firrtl.uint<1>
    %in = firrtl.instance r @Read(in in : !firrtl.probe<uint<1>>)
    firrtl.ref.define %in, %ref : !firrtl.probe<uint<1>>
  }
}

// -----

// U-turn: Can't use input probes.  Hoist-passthroughs is expected to handle this.
firrtl.circuit "UTurnTH" {
  firrtl.module private @UTurn(
                               // expected-note @below {{input probe}}
                               in %in : !firrtl.probe<uint<5>>,
                               // expected-error @below {{argument depends on input probe}}
                               out %out : !firrtl.probe<uint<5>>) {
    firrtl.ref.define %out, %in : !firrtl.probe<uint<5>>
  }
  firrtl.module @UTurnTH(in %in : !firrtl.uint<5>, out %out : !firrtl.uint<5>) {
    %u_in, %u_out  = firrtl.instance u @UTurn(in in : !firrtl.probe<uint<5>>,
                                              out out : !firrtl.probe<uint<5>>)
    %ref = firrtl.ref.send %in : !firrtl.uint<5>
    firrtl.ref.define %u_in, %ref : !firrtl.probe<uint<5>>
    %data = firrtl.ref.resolve %u_out : !firrtl.probe<uint<5>>
  }
}

