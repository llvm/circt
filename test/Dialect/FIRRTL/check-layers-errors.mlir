// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-layers))' %s --verify-diagnostics --split-input-file

firrtl.circuit "Simple" {
  firrtl.layer @A bind {}
  firrtl.module @Simple() {
    // expected-note @below {{enclosing bound layerblock here}}
    firrtl.layerblock @A {
      // expected-note @below {{instantiation under a bound layerblock here}}
      firrtl.instance layers @Layers()
    }
  }
  // expected-error @below {{module contains bound layer blocks and is instantiated under a bound layer block}}
  firrtl.module @Layers() {
    // expected-note @below {{bound child layerblock here}}
    firrtl.layerblock @A {}
  }
}

// -----

firrtl.circuit "Transitive" {
  firrtl.layer @A bind {}
  firrtl.module @Transitive() {
    // expected-note @below {{enclosing bound layerblock here}}
    firrtl.layerblock @A {
      // expected-note @below {{instantiation under a bound layerblock here}}
      firrtl.instance middle @Middle()
    }
  }
  firrtl.module @Middle() {
    // expected-note @below {{instantiation under a bound layerblock here}}
    firrtl.instance layers @Layers()
  }
  // expected-error @below {{module contains bound layer blocks and is instantiated under a bound layer block}}
  firrtl.module @Layers() {
    // expected-note @below {{bound child layerblock here}}
    firrtl.layerblock @A {}
  }
}

// -----

firrtl.circuit "FirstLayerBlockFound" {
  firrtl.layer @A bind {}
  firrtl.module @FirstLayerBlockFound() {
    // expected-note @below {{enclosing bound layerblock here}}
    firrtl.layerblock @A {
      // expected-note @below {{instantiation under a bound layerblock here}}
      firrtl.instance layers @Layers()
    }
  }
  // expected-error @below {{module contains bound layer blocks and is instantiated under a bound layer block}}
  firrtl.module @Layers() {
    // expected-note @below {{bound child layerblock here}}
    firrtl.layerblock @A {}
    // expected-note @below {{bound child layerblock here}}
    firrtl.layerblock @A {}
  }
}

// -----

firrtl.circuit "MultipleErrors" {
  firrtl.layer @A bind {}
  firrtl.module @MultipleErrors() {
    // expected-note @below {{enclosing bound layerblock here}}
    firrtl.layerblock @A {
      // expected-note @below {{instantiation under a bound layerblock here}}
      firrtl.instance layers1 @Layers1()
      // expected-note @below {{instantiation under a bound layerblock here}}
      firrtl.instance layers2 @Layers2()
    }
  }
  // expected-error @below {{module contains bound layer blocks and is instantiated under a bound layer block}}
  firrtl.module @Layers1() {
    // expected-note @below {{bound child layerblock here}}
    firrtl.layerblock @A {}
  }
  // expected-error @below {{module contains bound layer blocks and is instantiated under a bound layer block}}
  firrtl.module @Layers2() {
    // expected-note @below {{bound child layerblock here}}
    firrtl.layerblock @A {}
  }
}

// -----

firrtl.circuit "MultipleErrors" {
  firrtl.layer @A bind {}
  firrtl.module @MultipleErrors() {
    // expected-note @below {{enclosing bound layerblock here}}
    firrtl.layerblock @A {
      // expected-note @below {{instantiation under a bound layerblock here}}
      firrtl.instance layers1 @Layers()
    }
  }
  firrtl.module @OtherTop() {
    // expected-note @below {{enclosing bound layerblock here}}
    firrtl.layerblock @A {
      // expected-note @below {{instantiation under a bound layerblock here}}
      firrtl.instance layers1 @Layers()
    }
  }
  // expected-error @below {{module contains bound layer blocks and is instantiated under a bound layer block}}
  firrtl.module @Layers() {
    // expected-note @below {{bound child layerblock here}}
    firrtl.layerblock @A {}
  }
}

// -----

firrtl.circuit "NestedLayers" {
  firrtl.layer @A bind {}
  firrtl.module @NestedLayers() {
    // expected-note @below {{enclosing bound layerblock here}}
    firrtl.layerblock @A {
      // expected-note @below {{instantiation under a bound layerblock here}}
      firrtl.instance layera @LayerA()
    }
  }
  // expected-error @below {{module contains bound layer blocks and is instantiated under a bound layer block}}
  firrtl.module @LayerA() {
    // expected-note @+2 {{enclosing bound layerblock here}}
    // expected-note @+1 {{bound child layerblock here}}
    firrtl.layerblock @A {
      // expected-note @below {{instantiation under a bound layerblock here}}
      firrtl.instance layerb @LayerB()
    }
  }
  // expected-error @below {{module contains bound layer blocks and is instantiated under a bound layer block}}
  firrtl.module @LayerB() {
    // expected-note @below {{bound child layerblock here}}
    firrtl.layerblock @A {}
  }
}

// -----

firrtl.circuit "RegionOps" {
  firrtl.layer @A bind {}
  firrtl.module @RegionOps(in %in : !firrtl.uint<1>) {
    firrtl.when %in : !firrtl.uint<1> {
      // expected-note @below {{enclosing bound layerblock here}}
      firrtl.layerblock @A {
        // expected-note @below {{instantiation under a bound layerblock here}}
        %layers_in = firrtl.instance layers @Layers(in in : !firrtl.enum<a: uint<1>>)
      }
    }
  }
  // expected-error @below {{module contains bound layer blocks and is instantiated under a bound layer block}}
  firrtl.module @Layers(in %in : !firrtl.enum<a: uint<1>>) {
    firrtl.match %in : !firrtl.enum<a: uint<1>> {
      case a(%arg0) {
        // expected-note @below {{bound child layerblock here}}
        firrtl.layerblock @A {}
      }
    }
  }
}

firrtl.circuit "InstanceUnderWhen" {
  firrtl.layer @A bind {}
  firrtl.module @InstanceUnderWhen(in %in : !firrtl.uint<1>) {
    // expected-note @below {{enclosing bound layerblock here}}
    firrtl.layerblock @A {
      firrtl.when %in : !firrtl.uint<1> {
        // expected-note @below {{instantiation under a bound layerblock here}}
        %layers_in = firrtl.instance layers @Layers(in in : !firrtl.enum<a: uint<1>>)
      }
   }
  }

  // expected-error @below {{module contains bound layer blocks and is instantiated under a bound layer block}}
  firrtl.module @Layers(in %in : !firrtl.enum<a: uint<1>>) {
    // expected-note @below {{bound child layerblock here}}
    firrtl.layerblock @A {}
  }
}

// -----

// A Grand Central companion cannot contain layerblocks.
firrtl.circuit "Foo" {
  firrtl.layer @A bind {}
  // expected-error @below {{grand central companion module contains bound layerblocks}}
  firrtl.module @Bar() attributes {
    annotations = [
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "GroundView",
        id = 0 : i64,
        name = "GroundView"
      }
    ]
  } {
    // expected-note @below {{bound child layerblock here}}
    firrtl.layerblock @A {}
  }
  firrtl.module @Foo() {
    firrtl.instance bar @Bar()
  }
}

// -----

// A Grand Central companion cannot be instantiated under a layerblock.
firrtl.circuit "Foo" {
  firrtl.layer @A bind {}
  // expected-error @below {{grand central companion module is instantiated under a bound layerblock}}
  firrtl.module @Bar() attributes {
    annotations = [
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "GroundView",
        id = 0 : i64,
        name = "GroundView"
      }
    ]
  } {
  }
  firrtl.module @Foo() {
    // expected-note @below {{enclosing bound layerblock here}}
    firrtl.layerblock @A {
      // expected-note @below {{instantiation under a bound layerblock here}}
      firrtl.instance bar @Bar()
    }
  }
}

// -----

// A Grand Central companion cannot be instantiated under another companion.
firrtl.circuit "Foo" {
  firrtl.layer @A bind {}
  // expected-error @below {{grand central companion module is instantiated under another grand central companion module}}
  firrtl.module @Bar() attributes {
    annotations = [
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "GroundView",
        id = 0 : i64,
        name = "GroundView"
      }
    ]
  } {
  }
  // expected-note @below {{enclosing grand central companion module here}}
  firrtl.module @Baz() attributes {
    annotations = [
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "GroundView",
        id = 0 : i64,
        name = "GroundView"
      }
    ]
  } {
    // expected-note @below {{instantiation under a grand central companion module here}}
    firrtl.instance bar @Bar()
  }
  firrtl.module @Foo() {}
}
