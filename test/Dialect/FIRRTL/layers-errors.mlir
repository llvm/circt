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
  firrtl.extmodule @Foo(out o : !firrtl.probe<uint<1>, @B>) attributes {knownLayers =[@B]}
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

// -----

firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  firrtl.module @Foo() attributes {layers = [@A]} {}
  firrtl.module @Top() {
    // expected-error @below {{'firrtl.instance' op ambient layers are insufficient to instantiate module}}
    // expected-note  @below {{missing layer requirements: @A}}
    firrtl.instance foo {layers = [@A]} @Foo()
  }
}

// -----

firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  firrtl.option @O {
    firrtl.option_case @C1
  }
  firrtl.module @Foo() attributes {layers = [@A]} {}
  firrtl.module @Bar() attributes {layers = [@A]} {}
  firrtl.module @Top() {
    // expected-error @below {{'firrtl.instance_choice' op ambient layers are insufficient to instantiate module}}
    // expected-note  @below {{missing layer requirements: @A}}
    firrtl.instance_choice foo {layers = [@A]} @Foo alternatives @O {
      @C1 -> @Bar
    } ()
  }
}

// -----

// Capturing an outer property from inside a layerblock is not allowed.
// Eventually, we may like to relax this, but we need time to convince
// ourselves whether this is actually safe and can be lowered.
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  firrtl.extmodule @WithInputProp(in i : !firrtl.string) attributes {knownLayers=[@A], layers=[@A]}
  firrtl.module @Top(out %o : !firrtl.probe<uint<1>>) {
    // expected-note @below {{operand is defined here}}
    %str = firrtl.string "whatever"
    // expected-error @below {{'firrtl.layerblock' op captures a property operand}}
    firrtl.layerblock @A {
       %foo_in = firrtl.instance foo @WithInputProp(in i : !firrtl.string)
       // expected-note @below {{operand is used here}}
      firrtl.propassign %foo_in, %str : !firrtl.string
    }
  }
}

// -----

// Driving an outer property from inside a layerblock is not allowed.
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  firrtl.extmodule @WithInputProp(in i : !firrtl.string) attributes {knownLayers=[@A], layers=[@A]}
  firrtl.module @Top(out %o : !firrtl.probe<uint<1>>) {
    // expected-note @below {{operand is defined here}}
    %foo_in = firrtl.instance foo @WithInputProp(in i : !firrtl.string)
    // expected-error @below {{'firrtl.layerblock' op captures a property operand}}
    firrtl.layerblock @A {
      %str = firrtl.string "whatever"
       // expected-note @below {{operand is used here}}
      firrtl.propassign %foo_in, %str : !firrtl.string
    }
  }
}
