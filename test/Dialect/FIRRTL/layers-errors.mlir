// RUN: circt-opt %s -split-input-file -verify-diagnostics

// Cannot cast away a probe color.
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  firrtl.module @Top() {
    %w = firrtl.wire : !firrtl.probe<uint<1>, @A>
    // expected-error @below {{cannot discard layer requirements of input ref}}
    // expected-note  @below {{discarding layer requirements: @A}}
    %s = firrtl.ref.cast %w : (!firrtl.probe<uint<1>, @A>) -> !firrtl.probe<uint<1>>
  }
}

// -----

firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  firrtl.module @Top(out %o : !firrtl.probe<uint<1>>) {
    %c = firrtl.constant 0 : !firrtl.uint<1>
    %r = firrtl.ref.send %c : !firrtl.uint<1>
    firrtl.layerblock @A {
      // expected-error @below {{'firrtl.ref.define' op has more layer requirements than destination}}
      // expected-note  @below {{additional layers required: @A}}
      firrtl.ref.define %o, %r : !firrtl.probe<uint<1>>
    }
  }
}

// -----

firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  firrtl.module @Top(out %o : !firrtl.probe<uint<1>>) {
    firrtl.layerblock @A {
      %c = firrtl.constant 0  : !firrtl.uint<1>
      %r = firrtl.ref.send %c : !firrtl.uint<1>
      // expected-error @below {{'firrtl.ref.define' op has more layer requirements than destination}}
      // expected-note  @below {{additional layers required: @A}}
      firrtl.ref.define %o, %r : !firrtl.probe<uint<1>>
    }
  }
}

// -----

firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  firrtl.layer @B bind {}
  firrtl.extmodule @Foo(out o : !firrtl.probe<uint<1>, @B>)
  firrtl.module @Top(out %o : !firrtl.probe<uint<1>, @B>) {
    firrtl.layerblock @A {
      %foo_o = firrtl.instance foo @Foo(out o : !firrtl.probe<uint<1>, @B>)
      // expected-error @below {{'firrtl.ref.define' op has more layer requirements than destination}}
      // expected-note  @below {{additional layers required: @A}}
      firrtl.ref.define %o, %foo_o : !firrtl.probe<uint<1>, @B>
    }
  }
}

// -----

firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  firrtl.module @Top(out %o : !firrtl.probe<uint<1>>) {
    %w = firrtl.wire : !firrtl.openbundle<f: probe<uint<1>>>
    firrtl.layerblock @A {
      %f = firrtl.opensubfield %w[f] : !firrtl.openbundle<f: probe<uint<1>>>
      %c = firrtl.constant 0 : !firrtl.uint<1>
      %r = firrtl.ref.send %c : !firrtl.uint<1>
      // expected-error @below {{'firrtl.ref.define' op has more layer requirements than destination}}
      // expected-note  @below {{additional layers required: @A}}
      firrtl.ref.define %f, %r : !firrtl.probe<uint<1>>
    }
  }
}

// -----

// ref.resolve under insufficient layers.
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  firrtl.module @Top() {
    %w = firrtl.wire : !firrtl.probe<uint<1>, @A>
    // expected-error @below {{'firrtl.ref.resolve' op ambient layers are insufficient to resolve reference}}
    // expected-note  @below {{missing layer requirements: @A}}
    %0 = firrtl.ref.resolve %w : !firrtl.probe<uint<1>, @A>
  }
}

// -----

// ref.resolve under insufficient layers (with nested layers).
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  firrtl.module @Top() {
    %w = firrtl.wire : !firrtl.probe<uint<1>, @A::@B>
    firrtl.layerblock @A {
      // expected-error @below {{'firrtl.ref.resolve' op ambient layers are insufficient to resolve reference}}
      // expected-note  @below {{missing layer requirements: @A}}
      %0 = firrtl.ref.resolve %w : !firrtl.probe<uint<1>, @A::@B>
    }
  }
}
