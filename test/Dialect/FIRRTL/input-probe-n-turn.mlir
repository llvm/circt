// RUN: firtool %s --verify-diagnostics
// For now, input probes must be passthrough (or unused).

// Per 3.0.0 spec however, they should be supported as long
// as the input probe is equivalently driven in all contexts
// and resolves to a signal below its use.

// Arguably this should be XFAIL, but instead let's test the expected
// behavior where this is rejected with diagnostic.

firrtl.circuit "NTurn" {
  firrtl.extmodule @Ext(in sink: !firrtl.uint<1>,
                        out ref: !firrtl.probe<uint<1>>)
  firrtl.module private @UseAndExport(in %in: !firrtl.probe<uint<1>>,
                                      // expected-note @above {{input probe here}}
                                      out %out: !firrtl.probe<uint<1>>) {
    %sink, %ref = firrtl.instance e @Ext(in sink: !firrtl.uint<1>,
                                         out ref: !firrtl.probe<uint<1>>)
    firrtl.ref.define %out, %ref : !firrtl.probe<uint<1>>
    // expected-error @below {{input probes cannot be used}}
    %read = firrtl.ref.resolve %in : !firrtl.probe<uint<1>>
    firrtl.strictconnect %sink, %read : !firrtl.uint<1>
  }
  firrtl.module @NTurn() {
    %in, %out = firrtl.instance uae @UseAndExport(in in: !firrtl.probe<uint<1>>,
                                                  out out: !firrtl.probe<uint<1>>)

    // Send probe back in for use.
    firrtl.ref.define %in, %out : !firrtl.probe<uint<1>>
  }
}
