// RUN: circt-opt %s --split-input-file --firrtl-link-circuits="base-circuit=Foo" | FileCheck %s

// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" {
  firrtl.module @Foo() {}
  firrtl.layer @A0 bind {
    firrtl.layer @A0B0 bind {
      firrtl.layer @A0B0C0 bind {
      }
      firrtl.layer @A0B0C1 bind {
      }
    }
    firrtl.layer @A0B1 bind {
    }
    firrtl.layer @A0B2 bind {
    }
  }
  firrtl.layer @A1 bind {
  }
}

firrtl.circuit "Foo" {
  firrtl.extmodule @Foo()
  firrtl.layer @A0 bind {
    firrtl.layer @A0B0 bind {
      firrtl.layer @A0B0C1 bind {
      }
      firrtl.layer @A0B0C2 bind { 
      }
    }
    firrtl.layer @A0B3 bind {
    }
  }
  firrtl.layer @A1 bind {
    firrtl.layer @A1B0 bind {
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
