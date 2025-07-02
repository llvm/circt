// RUN: circt-opt %s

firrtl.circuit "Test" {

  firrtl.module @Test() {}

  firrtl.layer @A bind {
    firrtl.layer @B bind {
    }
  }

  firrtl.layer @B bind {}

  firrtl.module @Foo(out %o : !firrtl.probe<uint<1>, @A>) {
    %c = firrtl.constant 0 : !firrtl.uint<1>
    %r = firrtl.ref.send %c : !firrtl.uint<1>
    %s = firrtl.ref.cast %r : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>, @A>
    firrtl.ref.define %o, %s : !firrtl.probe<uint<1>, @A>
  }

  firrtl.module @Bar(out %o : !firrtl.probe<uint<1>, @B>) {
    %c = firrtl.constant 0 : !firrtl.uint<1>
    %r = firrtl.ref.send %c : !firrtl.uint<1>
    %s = firrtl.ref.cast %r : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>, @B>
    firrtl.ref.define %o, %s : !firrtl.probe<uint<1>, @B>
  }

  //===--------------------------------------------------------------------===//
  // Capture Tests
  //===--------------------------------------------------------------------===//

  firrtl.module @CaptureTests() {
    %a = firrtl.wire : !firrtl.vector<bundle<a: uint<1>, b flip: uint<1>>, 2>
    %b = firrtl.wire : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
    firrtl.layerblock @A {
      %0 = firrtl.subindex %a[0] : !firrtl.vector<bundle<a: uint<1>, b flip: uint<1>>, 2>
      %1 = firrtl.constant 0 : !firrtl.uint<1>
      %2 = firrtl.subaccess %a[%1] : !firrtl.vector<bundle<a: uint<1>, b flip: uint<1>>, 2>, !firrtl.uint<1>
      %3 = firrtl.subfield %b[a] : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
      %4 = firrtl.ref.send %b : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
    }
  }

  //===--------------------------------------------------------------------===//
  // Connect Tests
  //===--------------------------------------------------------------------===//

  firrtl.module @ConnectTests() {
    %a = firrtl.wire : !firrtl.uint<1>
    %b = firrtl.wire : !firrtl.bundle<a: uint<1>>
    %c = firrtl.wire : !firrtl.bundle<a flip: uint<1>>
    %d = firrtl.wire : !firrtl.bundle<a : bundle< a flip : uint<1>>>
    %e = firrtl.wire : !firrtl.bundle<a flip : bundle< a flip : uint<1>>>
    firrtl.layerblock @A {
      %_a = firrtl.wire : !firrtl.uint<1>
      %_b = firrtl.wire : !firrtl.bundle<a: uint<1>>
      %_c = firrtl.wire : !firrtl.bundle<a flip: uint<1>>
      %_d = firrtl.wire : !firrtl.bundle<a : bundle< a flip : uint<1>>>
      %_e = firrtl.wire : !firrtl.bundle<a flip : bundle< a flip : uint<1>>>

      firrtl.connect %_a, %a : !firrtl.uint<1>
      firrtl.connect %_b, %b : !firrtl.bundle<a: uint<1>>
      firrtl.connect %c, %_c : !firrtl.bundle<a flip: uint<1>>
      firrtl.connect %d, %_d : !firrtl.bundle<a : bundle< a flip : uint<1>>>
      firrtl.connect %_e, %e : !firrtl.bundle<a flip : bundle< a flip : uint<1>>>
    }
  }

  //===--------------------------------------------------------------------===//
  // Basic Casting Tests
  //===--------------------------------------------------------------------===//

  firrtl.module @CastToAnyLayer() {
    // Safe to take an uncolored probe and colorize it.
    %c = firrtl.constant 0 : !firrtl.uint<1>
    %r = firrtl.ref.send %c : !firrtl.uint<1>
    %s = firrtl.ref.cast %r : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>, @A>
  }

  firrtl.module @CastToSublayer() {
    // Safe to take a colored probe and cast it to a nested/child/sub-layer.
    %c = firrtl.constant 0 : !firrtl.uint<1>
    %r = firrtl.ref.send %c : !firrtl.uint<1>
    %s = firrtl.ref.cast %r : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>, @A>
    %t = firrtl.ref.cast %s : (!firrtl.probe<uint<1>, @A>) -> !firrtl.probe<uint<1>, @A::@B>
  }

  firrtl.module @CastToAmbientLayer(out %o : !firrtl.probe<uint<1>, @A::@B>) {
    // safe to take a value under layerblock @A and cast it to @A::B
    firrtl.layerblock @A {
      %c = firrtl.constant 0 : !firrtl.uint<1>
      %r = firrtl.ref.send %c : !firrtl.uint<1>
      %s = firrtl.ref.cast %r : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>, @A::@B>
      firrtl.ref.define %o, %s : !firrtl.probe<uint<1>, @A::@B>
    }
  }

  firrtl.module @CastAwayColorUnderLayerBlock(out %o : !firrtl.probe<uint<1>>) {
    %w = firrtl.wire : !firrtl.probe<uint<1>, @A>

    // define %w, a wire colored by @A
    firrtl.layerblock @A {
      %c = firrtl.constant 0 : !firrtl.uint<1>
      %r = firrtl.ref.send %c : !firrtl.uint<1>
    }

    // erase color from %w, safe under layerblock @A
    firrtl.layerblock @A {
      %r = firrtl.ref.cast %w : (!firrtl.probe<uint<1>, @A>) -> !firrtl.probe<uint<1>>
      %d = firrtl.ref.resolve %r : !firrtl.probe<uint<1>>
    }
  }

  firrtl.module @CastAwayPortColorUnderLayerBlock() {
    firrtl.layerblock @A {
      %o = firrtl.instance foo @Foo(out o : !firrtl.probe<uint<1>, @A>)
      // safe to cast away @A from %o under layerblock @A
      %r = firrtl.ref.cast %o : (!firrtl.probe<uint<1>, @A>) -> !firrtl.probe<uint<1>>
    }
  }

  firrtl.module @CastAwayColorUnderEnabledLayer(out %o : !firrtl.probe<uint<1>>) attributes {layers = [@A]} {
    %w = firrtl.wire : !firrtl.probe<uint<1>, @A>
    firrtl.layerblock @A {
      %c = firrtl.constant 0 : !firrtl.uint<1>
      %r = firrtl.ref.send %c : !firrtl.uint<1>
      %s = firrtl.ref.cast %w : (!firrtl.probe<uint<1>, @A>) -> !firrtl.probe<uint<1>>
      %d = firrtl.ref.resolve %s : !firrtl.probe<uint<1>>
    }
    // cast away the @A from %w, safe under enabled layers.
    %r = firrtl.ref.cast %w : (!firrtl.probe<uint<1>, @A>) -> !firrtl.probe<uint<1>>
    firrtl.ref.define %o, %r : !firrtl.probe<uint<1>>
  }

  //===--------------------------------------------------------------------===//
  // Enabled Layers Tests
  //===--------------------------------------------------------------------===//

  firrtl.module @CastAwayPortColorUnderEnabledLayer(out %o : !firrtl.probe<uint<1>>) attributes {layers = [@A]} {
    %foo_o = firrtl.instance foo @Foo(out o : !firrtl.probe<uint<1>, @A>)
    // safe to cast away @A from %foo_o under enabled layers
    %r = firrtl.ref.cast %foo_o : (!firrtl.probe<uint<1>, @A>) -> !firrtl.probe<uint<1>>
    firrtl.ref.define %o, %r : !firrtl.probe<uint<1>>
  }

  firrtl.module @DefineUncoloredProbeUnderEnabledLayerWithReferentUnderLayerBlock(out %o : !firrtl.probe<uint<1>>)
  attributes {layers = [@A]} {
    firrtl.layerblock @A {
      %c = firrtl.constant 0 : !firrtl.uint<1>
      %r = firrtl.ref.send %c : !firrtl.uint<1>
      firrtl.ref.define %o, %r : !firrtl.probe<uint<1>>
    }
  }

  firrtl.module @CastAwayMultipleColorsUnderEnabledLayers(out %o : !firrtl.probe<uint<1>>)
  attributes {layers = [@A, @B]} {
    firrtl.layerblock @A {
      %bar_o = firrtl.instance bar @Bar(out o : !firrtl.probe<uint<1>, @B>)
      %r = firrtl.ref.cast %bar_o : (!firrtl.probe<uint<1>, @B>) -> !firrtl.probe<uint<1>>
      firrtl.ref.define %o, %r : !firrtl.probe<uint<1>>
    }
  }

  //===--------------------------------------------------------------------===//
  // Ref Resolve Tests
  //===--------------------------------------------------------------------===//

  firrtl.module @ResolveColoredRefUnderLayerBlock() {
    %w = firrtl.wire : !firrtl.probe<uint<1>, @A>
    firrtl.layerblock @A {
      %0 = firrtl.ref.resolve %w : !firrtl.probe<uint<1>, @A>
    }
  }

  firrtl.module @ResolveColoredRefUnderEnabledLayer() attributes {layers=[@A]} {
    %w = firrtl.wire : !firrtl.probe<uint<1>, @A>
    %0 = firrtl.ref.resolve %w : !firrtl.probe<uint<1>, @A>
  }

  firrtl.module @ResolveColoredRefPortUnderLayerBlock1() {
    %foo_o = firrtl.instance foo @Foo(out o : !firrtl.probe<uint<1>, @A>)
    firrtl.layerblock @A {
      %x = firrtl.ref.resolve %foo_o : !firrtl.probe<uint<1>, @A>
    }
  }

  firrtl.module @ResolveColoredRefPortUnderLayerBlock2() {
    firrtl.layerblock @A {
      %foo_o = firrtl.instance foo @Foo(out o : !firrtl.probe<uint<1>, @A>)
      %x = firrtl.ref.resolve %foo_o : !firrtl.probe<uint<1>, @A>
    }
  }

  firrtl.module @ResolveColoredRefPortUnderEnabledLayer() attributes {layers=[@A]} {
    %foo_o = firrtl.instance foo @Foo(out o : !firrtl.probe<uint<1>, @A>)
    %x = firrtl.ref.resolve %foo_o : !firrtl.probe<uint<1>, @A>
  }

  //===--------------------------------------------------------------------===//
  // Regression Tests
  //===--------------------------------------------------------------------===//

  firrtl.module @WhenUnderLayer(in %test: !firrtl.uint<1>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.layerblock @A {
      %w = firrtl.wire : !firrtl.uint<1>
      firrtl.when %test : !firrtl.uint<1> {
        firrtl.matchingconnect %w, %c0_ui1 : !firrtl.uint<1>
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

  //===--------------------------------------------------------------------===//
  // Properties Under Layers
  //===--------------------------------------------------------------------===//

  firrtl.extmodule @WithInputProp(in i : !firrtl.string)

  firrtl.module @InstWithInputPropUnderLayerBlock() {
    firrtl.layerblock @A {
      %foo_in = firrtl.instance foo @WithInputProp(in i : !firrtl.string)
      %str = firrtl.string "whatever"
      firrtl.propassign %foo_in, %str : !firrtl.string
    }
  }
}
