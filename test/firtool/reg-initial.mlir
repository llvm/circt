// RUN: firtool %s --format=mlir --split-input-file | FileCheck %s

// An end-to-end test of the `initial` time-zero simulation value attribute on
// `firrtl.reg` and `firrtl.regreset`, lowered all the way to Verilog by
// firtool. The value appears as an inline `sv.reg` initializer with no
// `initial` block or ifdef guard.

firrtl.circuit "RegInitial" {
  firrtl.module @RegInitial(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                            in %d: !firrtl.uint<8>, out %q: !firrtl.uint<8>,
                            out %o: !firrtl.uint<8>) {
    // CHECK: reg [7:0] r = 8'h5;
    %r = firrtl.reg %clock {initial = 5 : ui8} : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>

    %c7 = firrtl.constant 7 : !firrtl.uint<8>
    // CHECK: reg [7:0] s = 8'h0;
    %s = firrtl.regreset %clock, %reset, %c7 {initial = 0 : ui8} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.matchingconnect %s, %d : !firrtl.uint<8>

    firrtl.matchingconnect %q, %r : !firrtl.uint<8>
    firrtl.matchingconnect %o, %s : !firrtl.uint<8>
  }
}

// -----

// A preset register buried under an inline layer (which lowers to an
// `` `ifdef ``) is emitted with an inline initializer inside that same guard,
// with no `initial` block or XMR for the preset.

firrtl.circuit "RegInitialBuried" {
  firrtl.layer @A inline { }
  firrtl.module @RegInitialBuried(in %clock: !firrtl.clock, in %d: !firrtl.uint<8>,
                                  out %probe: !firrtl.probe<uint<8>, @A>) {
    // CHECK: `ifdef layer$A
    // CHECK:   reg [7:0] r = 8'h5;
    // CHECK:   always @(posedge clock)
    // CHECK:     r <= d;
    // CHECK: `endif // layer$A
    // CHECK-NOT: initial
    firrtl.layerblock @A {
      %r = firrtl.reg %clock {initial = 5 : ui8} : !firrtl.clock, !firrtl.uint<8>
      firrtl.matchingconnect %r, %d : !firrtl.uint<8>
      %ref = firrtl.ref.send %r : !firrtl.uint<8>
      %cast = firrtl.ref.cast %ref : (!firrtl.probe<uint<8>>) -> !firrtl.probe<uint<8>, @A>
      firrtl.ref.define %probe, %cast : !firrtl.probe<uint<8>, @A>
    }
  }
}
