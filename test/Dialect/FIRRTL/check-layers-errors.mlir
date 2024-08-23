// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-layers))' %s --verify-diagnostics --split-input-file

firrtl.circuit "Simple" {
  firrtl.layer @A bind {}
  firrtl.module @Simple() {
    firrtl.layerblock @A {
      // expected-error @below {{cannot instantiate "Layers" under a layerblock, because "Layers" contains a layerblock}}
      firrtl.instance layers @Layers()
    }
  }
  firrtl.module @Layers() {
    // expected-note @below {{layerblock here}}
    firrtl.layerblock @A {}
  }
}

// -----

firrtl.circuit "Transitive" {
  firrtl.layer @A bind {}
  firrtl.module @Transitive() {
    firrtl.layerblock @A {
      // expected-error @below {{cannot instantiate "Middle" under a layerblock, because "Middle" contains a layerblock}}
      firrtl.instance middle @Middle()
    }
  }
  firrtl.module @Middle() {
    firrtl.instance layers @Layers()
  }
  firrtl.module @Layers() {
    // expected-note @below {{layerblock here}}
    firrtl.layerblock @A {}
  }
}

// -----

firrtl.circuit "FirstLayerBLockFound" {
  firrtl.layer @A bind {}
  firrtl.module @FirstLayerBLockFound() {
    firrtl.layerblock @A {
      // expected-error @below {{cannot instantiate "Layers" under a layerblock, because "Layers" contains a layerblock}}
      firrtl.instance layers @Layers()
    }
  }
  firrtl.module @Layers() {
    // expected-note @below {{layerblock here}}
    firrtl.layerblock @A {}
    firrtl.layerblock @A {}
  }
}

// -----

firrtl.circuit "MultipleErrors" {
  firrtl.layer @A bind {}
  firrtl.module @MultipleErrors() {
    firrtl.layerblock @A {
      // expected-error @below {{cannot instantiate "Layers1" under a layerblock, because "Layers1" contains a layerblock}}
      firrtl.instance layers1 @Layers1()
      // expected-error @below {{cannot instantiate "Layers2" under a layerblock, because "Layers2" contains a layerblock}}
      firrtl.instance layers2 @Layers2()
    }
  }
  firrtl.module @Layers1() {
    // expected-note @below {{layerblock here}}
    firrtl.layerblock @A {}
  }
  firrtl.module @Layers2() {
    // expected-note @below {{layerblock here}}
    firrtl.layerblock @A {}
  }
}

// -----

firrtl.circuit "MultipleErrors" {
  firrtl.layer @A bind {}
  firrtl.module @MultipleErrors() {
    firrtl.layerblock @A {
      // expected-error @below {{cannot instantiate "Layers" under a layerblock, because "Layers" contains a layerblock}}
      firrtl.instance layers1 @Layers()
    }
  }
  firrtl.module @OtherTop() {
    firrtl.layerblock @A {
      // expected-error @below {{cannot instantiate "Layers" under a layerblock, because "Layers" contains a layerblock}}
      firrtl.instance layers1 @Layers()
    }
  }
  firrtl.module @Layers() {
    // expected-note @+1 {{layerblock here}}
    firrtl.layerblock @A {}
  }
}

// -----

firrtl.circuit "NestedLayers" {
  firrtl.layer @A bind {}
  firrtl.module @NestedLayers() {
    firrtl.layerblock @A {
      // expected-error @below {{cannot instantiate "LayerA" under a layerblock, because "LayerA" contains a layerblock}}
      firrtl.instance layera @LayerA()
    }
  }
  firrtl.module @LayerA() {
    // expected-note @below {{layerblock here}}
    firrtl.layerblock @A {
      // expected-error @below {{cannot instantiate "LayerB" under a layerblock, because "LayerB" contains a layerblock}}
      firrtl.instance layerb @LayerB()
    }
  }
  firrtl.module @LayerB() {
    // expected-note @below {{layerblock here}}
    firrtl.layerblock @A {}
  }
}

// -----

firrtl.circuit "RegionOps" {
  firrtl.layer @A bind {}
  firrtl.module @RegionOps(in %in : !firrtl.uint<1>) {
    firrtl.when %in : !firrtl.uint<1> {
      firrtl.layerblock @A {
        // expected-error @below {{cannot instantiate "Layers" under a layerblock, because "Layers" contains a layerblock}}
        %layers_in = firrtl.instance layers @Layers(in in : !firrtl.enum<a: uint<1>>)
      }
    }
  }
  firrtl.module @Layers(in %in : !firrtl.enum<a: uint<1>>) {
    firrtl.match %in : !firrtl.enum<a: uint<1>> {
      case a(%arg0) {
        // expected-note @below {{layerblock here}}
        firrtl.layerblock @A {}
      }
    }
  }
}
