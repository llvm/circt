// RUN: circt-opt %s --split-input-file --firrtl-link-circuits="base-circuit=Foo" | FileCheck %s

// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" {
  firrtl.module @Foo() {}
  firrtl.layer @A0 bind {
    firrtl.layer @A0B0 bind {
      firrtl.layer @A0B0C0 bind {}
      firrtl.layer @A0B0C1 bind {}
    }
    firrtl.layer @A0B1 bind {}
    firrtl.layer @A0B2 bind {}
  }
  firrtl.layer @A1 bind {
  }
}

firrtl.circuit "Bar" {
  firrtl.module @Bar() {}
  firrtl.extmodule @Foo()
  firrtl.layer @A1 bind {
    firrtl.layer @A1B0 bind {}
  }
  firrtl.layer @A0 bind {
    firrtl.layer @A0B0 bind {
      firrtl.layer @A0B0C2 bind { 
      }
      firrtl.layer @A0B0C1 bind {
      }
    }
    firrtl.layer @A0B3 bind {
    }
  }
}

// CHECK:      firrtl.layer @A0 bind {
// CHECK-NEXT:   firrtl.layer @A0B0 bind {
// CHECK-NEXT:     firrtl.layer @A0B0C0 bind {
// CHECK-NEXT:     }
// CHECK-NEXT:     firrtl.layer @A0B0C1 bind {
// CHECK-NEXT:     }
// CHECK-NEXT:     firrtl.layer @A0B0C2 bind {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   firrtl.layer @A0B1 bind {
// CHECK-NEXT:   }
// CHECK-NEXT:   firrtl.layer @A0B2 bind {
// CHECK-NEXT:   }
// CHECK-NEXT:   firrtl.layer @A0B3 bind {
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: firrtl.layer @A1 bind {
// CHECK-NEXT:   firrtl.layer @A1B0 bind {
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" {
  firrtl.extmodule @Bar() attributes {knownLayers = [@A::@C]}
  firrtl.layer @A bind {
    firrtl.layer @B bind {}
    firrtl.layer @C bind {}
  }
  firrtl.module @Foo() {
    firrtl.instance inner @Bar()
  }
}

firrtl.circuit "Bar" {
  firrtl.module @Bar() {}
  firrtl.layer @A bind {
    firrtl.layer @C bind {}
  }
}

// CHECK:      firrtl.layer @A bind {
// CHECK-NEXT:   firrtl.layer @B bind {
// CHECK-NEXT:   }
// CHECK-NEXT:   firrtl.layer @C bind {
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" {
  firrtl.extmodule @Middle() attributes {knownLayers = [@L]}
  firrtl.layer @L bind {
    firrtl.layer @Top bind {}
  }
  firrtl.module @Foo() {
    firrtl.instance middle @Middle()
  }
}

firrtl.circuit "Middle" {
  firrtl.extmodule @Bottom() attributes {knownLayers = [@L]}
  firrtl.layer @L bind {
    firrtl.layer @Mid bind {}
  }
  firrtl.module @Middle() {
    firrtl.instance bottom @Bottom()
  }
}

firrtl.circuit "Bottom" {
  firrtl.module @Bottom() {}
  firrtl.layer @L bind {
    firrtl.layer @Bot bind {}
  }
}

// CHECK:      firrtl.layer @L bind {
// CHECK-NEXT:   firrtl.layer @Top bind {
// CHECK-NEXT:   }
// CHECK-NEXT:   firrtl.layer @Mid bind {
// CHECK-NEXT:   }
// CHECK-NEXT:   firrtl.layer @Bot bind {
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// CHECK-LABEL: firrtl.circuit "Foo"
// Test case: verify that multiple known layers are correctly checked and merged.
firrtl.circuit "Foo" {
  // Declaration requires both @A and @B
  firrtl.extmodule @Bar() attributes {knownLayers = [@A, @B]}
  firrtl.layer @B inline {}
  firrtl.layer @A inline {}
  firrtl.module @Foo() {
    firrtl.instance w @Bar()
  }
}

firrtl.circuit "Bar" {
  // Definition provides both @A and @B
  firrtl.module @Bar() {}
  firrtl.layer @A inline {
    firrtl.layer @ChildA inline {}
  }
  firrtl.layer @B inline {
    firrtl.layer @ChildB inline {}
  }
}

// CHECK-NEXT: firrtl.layer @B inline {
// CHECK-NEXT:   firrtl.layer @ChildB inline {
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK:      firrtl.layer @A inline {
// CHECK-NEXT:   firrtl.layer @ChildA inline {
// CHECK-NEXT:   }
// CHECK-NEXT: }
