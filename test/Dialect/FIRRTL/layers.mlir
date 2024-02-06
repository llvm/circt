// RUN: circt-opt %s

firrtl.circuit "Test" {
  firrtl.module @Test() {}

  firrtl.layer @A bind {}

  firrtl.module @WhenUnderLayer(in %test: !firrtl.uint<1>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.layerblock @A {
      %w = firrtl.wire : !firrtl.uint<1>
      firrtl.when %test : !firrtl.uint<1> {
        firrtl.strictconnect %w, %c0_ui1 : !firrtl.uint<1>
      }
    }
  }

  firrtl.module @ProbeEscapeLayer(out %p: !firrtl.probe<uint<1>, @A>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.layerblock @A {
      %0 = firrtl.ref.send %c0_ui1 : !firrtl.uint<1>
      %1 = firrtl.ref.cast %0 : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>, @A>
      firrtl.ref.define %p, %1 : !firrtl.probe<uint<1>, @A>
    }
  }

  firrtl.module @ProbeIntoOpenBundle(out %o: !firrtl.openbundle<p: probe<uint<1>, @A>>) {
    firrtl.layerblock @A {
      %0 = firrtl.opensubfield %o[p] : !firrtl.openbundle<p: probe<uint<1>, @A>>
      %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
      %1 = firrtl.ref.send %c0_ui1 : !firrtl.uint<1>
      %2 = firrtl.ref.cast %1 : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>, @A>
      firrtl.ref.define %0, %2 : !firrtl.probe<uint<1>, @A>
    }
  }
}
