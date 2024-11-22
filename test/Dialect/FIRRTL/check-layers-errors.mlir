// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-layers))' %s --verify-diagnostics --split-input-file

firrtl.circuit "Simple" {
  firrtl.layer @A bind {}
  firrtl.module @Simple() {
    firrtl.layerblock @A {
      // expected-note @below {{illegal instantiation under a layerblock here}}
      firrtl.instance layers @Layers()
    }
  }
  // expected-error @below {{either contains layerblocks or has at least one instance that is or contains a Grand Central companion or layerblocks}}
  firrtl.module @Layers() {
    // expected-note @below {{illegal layerblock here}}
    firrtl.layerblock @A {}
  }
}

// -----

firrtl.circuit "Transitive" {
  firrtl.layer @A bind {}
  firrtl.module @Transitive() {
    firrtl.layerblock @A {
      // expected-note @below {{illegal instantiation under a layerblock here}}
      firrtl.instance middle @Middle()
    }
  }
  // expected-error @below {{either contains layerblocks or has at least one instance that is or contains a Grand Central companion or layerblocks}}
  firrtl.module @Middle() {
    // expected-note @below {{illegal instantiation in a module under a layer here}}
    firrtl.instance layers @Layers()
  }
  // expected-error @below {{either contains layerblocks or has at least one instance that is or contains a Grand Central companion or layerblocks}}
  firrtl.module @Layers() {
    // expected-note @below {{illegal layerblock here}}
    firrtl.layerblock @A {}
  }
}

// -----

firrtl.circuit "FirstLayerBLockFound" {
  firrtl.layer @A bind {}
  firrtl.module @FirstLayerBLockFound() {
    firrtl.layerblock @A {
      // expected-note @below {{illegal instantiation under a layerblock here}}
      firrtl.instance layers @Layers()
    }
  }
  // expected-error @below {{either contains layerblocks or has at least one instance that is or contains a Grand Central companion or layerblocks}}
  firrtl.module @Layers() {
    // expected-note @below {{illegal layerblock here}}
    firrtl.layerblock @A {}
    // expected-note @below {{illegal layerblock here}}
    firrtl.layerblock @A {}
  }
}

// -----

firrtl.circuit "MultipleErrors" {
  firrtl.layer @A bind {}
  firrtl.module @MultipleErrors() {
    firrtl.layerblock @A {
      // expected-note @below {{illegal instantiation under a layerblock here}}
      firrtl.instance layers1 @Layers1()
      // expected-note @below {{illegal instantiation under a layerblock here}}
      firrtl.instance layers2 @Layers2()
    }
  }
  // expected-error @below {{either contains layerblocks or has at least one instance that is or contains a Grand Central companion or layerblocks}}
  firrtl.module @Layers1() {
    // expected-note @below {{illegal layerblock here}}
    firrtl.layerblock @A {}
  }
  // expected-error @below {{either contains layerblocks or has at least one instance that is or contains a Grand Central companion or layerblocks}}
  firrtl.module @Layers2() {
    // expected-note @below {{illegal layerblock here}}
    firrtl.layerblock @A {}
  }
}


// -----

firrtl.circuit "MultipleErrors" {
  firrtl.layer @A bind {}
  firrtl.module @MultipleErrors() {
    firrtl.layerblock @A {
      // expected-note @below {{illegal instantiation under a layerblock here}}
      firrtl.instance layers1 @Layers()
    }
  }
  firrtl.module @OtherTop() {
    firrtl.layerblock @A {
      // expected-note @below {{illegal instantiation under a layerblock here}}
      firrtl.instance layers1 @Layers()
    }
  }
  // expected-error @below {{either contains layerblocks or has at least one instance that is or contains a Grand Central companion or layerblocks}}
  firrtl.module @Layers() {
    // expected-note @below {{illegal layerblock here}}
    firrtl.layerblock @A {}
  }
}

// -----

firrtl.circuit "NestedLayers" {
  firrtl.layer @A bind {}
  firrtl.module @NestedLayers() {
    firrtl.layerblock @A {
      // expected-note @below {{illegal instantiation under a layerblock here}}
      firrtl.instance layera @LayerA()
    }
  }
  // expected-error @below {{either contains layerblocks or has at least one instance that is or contains a Grand Central companion or layerblocks}}
  firrtl.module @LayerA() {
    // expected-note @below {{illegal layerblock here}}
    firrtl.layerblock @A {
      // expected-note @below {{illegal instantiation under a layerblock here}}
      firrtl.instance layerb @LayerB()
    }
  }
  // expected-error @below {{either contains layerblocks or has at least one instance that is or contains a Grand Central companion or layerblocks}}
  firrtl.module @LayerB() {
    // expected-note @below {{illegal layerblock here}}
    firrtl.layerblock @A {}
  }
}

// -----

firrtl.circuit "RegionOps" {
  firrtl.layer @A bind {}
  firrtl.module @RegionOps(in %in : !firrtl.uint<1>) {
    firrtl.when %in : !firrtl.uint<1> {
      firrtl.layerblock @A {
        // expected-note @below {{illegal instantiation under a layerblock here}}
        %layers_in = firrtl.instance layers @Layers(in in : !firrtl.enum<a: uint<1>>)
      }
    }
  }
  // expected-error @below {{either contains layerblocks or has at least one instance that is or contains a Grand Central companion or layerblocks}}
  firrtl.module @Layers(in %in : !firrtl.enum<a: uint<1>>) {
    firrtl.match %in : !firrtl.enum<a: uint<1>> {
      case a(%arg0) {
        // expected-note @below {{illegal layerblock here}}
        firrtl.layerblock @A {}
      }
    }
  }
}

// -----

// A Grand Central companion cannot contain layerblocks.
firrtl.circuit "Foo" {
  firrtl.layer @A bind {}
  // expected-error @below {{is a Grand Central companion that either contains layerblocks or has at least one instance that is or contains a Grand Central companion or layerblocks}}
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
    // expected-note @below {{illegal layerblock here}}
    firrtl.layerblock @A {}
  }
  firrtl.module @Foo() {
    firrtl.instance bar @Bar()
  }
}

// -----

// A Grand Central companion cannot contain layerblocks.
firrtl.circuit "Foo" {
  firrtl.layer @A bind {}
  // expected-error @below {{is a Grand Central companion that either contains layerblocks or has at least one instance that is or contains a Grand Central companion or layerblocks}}
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
    firrtl.layerblock @A {
      // expected-note @below {{illegal instantiation under a layerblock here}}
      firrtl.instance bar @Bar()
    }
  }
}
