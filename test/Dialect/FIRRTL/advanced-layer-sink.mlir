// RUN: circt-opt -pass-pipeline="builtin.module(firrtl.circuit(firrtl-advanced-layer-sink))" -allow-unregistered-dialect %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Movement of layers to the back of their containing block.
//===----------------------------------------------------------------------===//

// using interesting_name to prevent layer-sink from deleting unused wires.

firrtl.circuit "Top" {
  // CHECK: firrtl.module @Top() {
  // CHECK: }
  firrtl.module @Top() {}
}

// Empty layerblock.
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  // CHECK: firrtl.module @Top() {
  // CHECK:   firrtl.layerblock @A {
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top() {
    firrtl.layerblock @A {}
  }
}

// Empty layerblock in layerblock.
firrtl.circuit "Top" {
  firrtl.layer @A bind {
    firrtl.layer @B bind {}
  }
  // CHECK: firrtl.module @Top() {
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     firrtl.layerblock @A::@B {
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top() {
    firrtl.layerblock @A {
      firrtl.layerblock @A::@B {
      }
    }
  }
}

// Layerblock at end already.
firrtl.circuit "Top" {
  firrtl.layer @A bind {
    firrtl.layer @B bind {}
  }
  // CHECK: firrtl.module @Top() {
  // CHECK:   %w = firrtl.wire interesting_name : !firrtl.uint<1>
  // CHECK:   firrtl.layerblock @A {
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top() {
    %w = firrtl.wire interesting_name : !firrtl.uint<1>
    firrtl.layerblock @A {
    }
  }
}

// Layerblock NOT at end already.
firrtl.circuit "Top" {
  firrtl.layer @A bind {
  }
  // CHECK: firrtl.module @Top() {
  // CHECK:   %w = firrtl.wire interesting_name : !firrtl.uint<1>
  // CHECK:   firrtl.layerblock @A {
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top() {
    firrtl.layerblock @A {
    }
    %w = firrtl.wire interesting_name : !firrtl.uint<1>
  }
}

// Parent layerblock at end already, nested layerblock NOT at end.
firrtl.circuit "Top" {
  firrtl.layer @A bind {
    firrtl.layer @B bind {}
  }
  // CHECK: firrtl.module @Top() {
  // CHECK:   %w1 = firrtl.wire interesting_name : !firrtl.uint<1>
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     %w2 = firrtl.wire interesting_name : !firrtl.uint<1>
  // CHECK:     firrtl.layerblock @A::@B {
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top() {
    %w1 = firrtl.wire interesting_name : !firrtl.uint<1>
    firrtl.layerblock @A {
      firrtl.layerblock @A::@B {
      }
      %w2 = firrtl.wire interesting_name : !firrtl.uint<1>
    }
  }
}

// Parent layerblock NOT at end already, nested layerblock NOT at end.
firrtl.circuit "Top" {
  firrtl.layer @A bind {
    firrtl.layer @B bind {}
  }
  // CHECK: firrtl.module @Top() {
  // CHECK:   %w1 = firrtl.wire interesting_name : !firrtl.uint<1>
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     %w2 = firrtl.wire interesting_name : !firrtl.uint<1>
  // CHECK:     firrtl.layerblock @A::@B {
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top() {
    firrtl.layerblock @A {
      firrtl.layerblock @A::@B {
      }
      %w2 = firrtl.wire interesting_name : !firrtl.uint<1>
    }
    %w1 = firrtl.wire interesting_name : !firrtl.uint<1>
  }
}

// Moving a layer past an op preserves the ordering of layerblocks.
firrtl.circuit "Top" {
  firrtl.layer @A bind {
    firrtl.layer @A1 bind {}
    firrtl.layer @A2 bind {}
    firrtl.layer @A3 bind {}
  }
  firrtl.layer @B bind {}
  firrtl.layer @C bind {}
  // CHECK: firrtl.module @Top() {
  // CHECK:   %w1 = firrtl.wire interesting_name : !firrtl.uint<1>
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     %w2 = firrtl.wire interesting_name : !firrtl.uint<1>
  // CHECK:     firrtl.layerblock @A::@A1 {
  // CHECK:     }
  // CHECK:     firrtl.layerblock @A::@A2 {
  // CHECK:     }
  // CHECK:     firrtl.layerblock @A::@A3 {
  // CHECK:     }
  // CHECK:   }
  // CHECK:   firrtl.layerblock @B {
  // CHECK:   }
  // CHECK:   firrtl.layerblock @C {
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top() {
    firrtl.layerblock @A {
      firrtl.layerblock @A::@A1 {
      }
      firrtl.layerblock @A::@A2 {
      }
      %w2 = firrtl.wire interesting_name : !firrtl.uint<1>
      firrtl.layerblock @A::@A3 {
      }
    }
   firrtl.layerblock @B {
    }
    %w1 = firrtl.wire interesting_name : !firrtl.uint<1>
    firrtl.layerblock @C {
    }
  }
}

// Parent layerblock NOT at end already, nested layerblock NOT at end.
firrtl.circuit "Top" {
  firrtl.layer @A bind {
    firrtl.layer @B bind {}
  }
  // CHECK: firrtl.module @Top() {
  // CHECK:   %w1 = firrtl.wire interesting_name : !firrtl.uint<1>
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     %w2 = firrtl.wire interesting_name : !firrtl.uint<1>
  // CHECK:     firrtl.layerblock @A::@B {
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top() {
    firrtl.layerblock @A {
      firrtl.layerblock @A::@B {
      }
      %w2 = firrtl.wire interesting_name : !firrtl.uint<1>
    }
    %w1 = firrtl.wire interesting_name : !firrtl.uint<1>
  }
}

//===----------------------------------------------------------------------===//
// Basic Sinking.
//===----------------------------------------------------------------------===//

// Sink a chain of expressions.
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  // CHECK: firrtl.module @Top() {
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:     %0 = firrtl.not %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK:     %node = firrtl.node %0 : !firrtl.uint<1>
  // CHECK:     "unknown"(%node) : (!firrtl.uint<1>) -> ()
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.not %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %node = firrtl.node %0 : !firrtl.uint<1>
    firrtl.layerblock @A {
      "unknown"(%node) : (!firrtl.uint<1>) -> ()
    }
  }
}

// Sink a chain of expressions rooted at an input port.
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  // CHECK: firrtl.module @Top(in %port: !firrtl.uint<1>) { 
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     %0 = firrtl.not %port : (!firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK:     %node = firrtl.node %0 : !firrtl.uint<1>
  // CHECK:     "unknown"(%node) : (!firrtl.uint<1>) -> ()
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top(in %port : !firrtl.uint<1>) {
    %0 = firrtl.not %port : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %node = firrtl.node %0 : !firrtl.uint<1>
    firrtl.layerblock @A {
      "unknown"(%node) : (!firrtl.uint<1>) -> ()
    }
  }
}

// Sink a node to the LCA of its uses.
firrtl.circuit "Top" {
  firrtl.layer @A bind {
    firrtl.layer @A1 bind {}
    firrtl.layer @A2 bind {}
  }
  // CHECK: firrtl.module @Top(in %port: !firrtl.uint<1>) {
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     %node = firrtl.node %port : !firrtl.uint<1>
  // CHECK:     firrtl.layerblock @A::@A1 {
  // CHECK:       "unknown"(%node) : (!firrtl.uint<1>) -> ()
  // CHECK:     }
  // CHECK:     firrtl.layerblock @A::@A2 {
  // CHECK:       "unknown"(%node) : (!firrtl.uint<1>) -> ()
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top(in %port: !firrtl.uint<1>) {
    %node = firrtl.node %port : !firrtl.uint<1>
    firrtl.layerblock @A {
      firrtl.layerblock @A::@A1 {
        "unknown"(%node) : (!firrtl.uint<1>) -> ()
      }
      firrtl.layerblock @A::@A2 {
        "unknown"(%node) : (!firrtl.uint<1>) -> ()
      }
    }
  }
}

// Output ports are demands: The unknown op should drag %node into layerblock
// @A, but the connection to %port, which uses %node as a source, will prevent
// the sinking of node into the layerblock.
firrtl.circuit "Top" {
 firrtl.layer @A bind {}
 firrtl.module @Top(out %port: !firrtl.uint<1>) {
   %c = firrtl.constant 0 : !firrtl.uint<1>
   %0 = firrtl.not %port : (!firrtl.uint<1>) -> !firrtl.uint<1>
   %node = firrtl.node %c : !firrtl.uint<1>
   firrtl.connect %port, %node : !firrtl.uint<1>, !firrtl.uint<1>
   firrtl.layerblock @A {
     "unknown"(%node) : (!firrtl.uint<1>) -> ()
   }
 }
}

//===----------------------------------------------------------------------===//
// Sinking Loops.
//===----------------------------------------------------------------------===//

// Sink a loop between two wires.
firrtl.circuit "Top" {
 firrtl.layer @A bind {}
  // CHECK: firrtl.module @Top() {
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     %w1 = firrtl.wire : !firrtl.uint<1>
  // CHECK:     %w2 = firrtl.wire : !firrtl.uint<1>
  // CHECK:     firrtl.matchingconnect %w1, %w2 : !firrtl.uint<1>
  // CHECK:     firrtl.matchingconnect %w2, %w1 : !firrtl.uint<1>
  // CHECK:     "unknown"(%w2) : (!firrtl.uint<1>) -> ()
  // CHECK:   }
  // CHECK: }
 firrtl.module @Top() {
   %w1 = firrtl.wire : !firrtl.uint<1>
   %w2 = firrtl.wire : !firrtl.uint<1>
   firrtl.matchingconnect %w1, %w2 : !firrtl.uint<1>
   firrtl.matchingconnect %w2, %w1 : !firrtl.uint<1>
   firrtl.layerblock @A {
     "unknown"(%w2) : (!firrtl.uint<1>) -> ()
   }
 }
}

// Sink a self-connected register.
firrtl.circuit "Top" {
 firrtl.layer @A bind {}
  // CHECK: firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:     %r = firrtl.regreset %clock, %reset, %c0_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:     firrtl.matchingconnect %r, %r : !firrtl.uint<1>
  // CHECK:     "unknown"(%r) : (!firrtl.uint<1>) -> ()
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %r = firrtl.regreset %clock, %reset, %c0_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.layerblock @A {
      "unknown"(%r) : (!firrtl.uint<1>) -> ()
    }
    firrtl.matchingconnect %r, %r : !firrtl.uint<1>
  }
}

//===----------------------------------------------------------------------===//
// Sinking Instances.
//===----------------------------------------------------------------------===//

// Pure instances can be moved.
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  firrtl.module @Pure(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
    firrtl.matchingconnect %o, %i : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @Top() {
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:     %pure_i, %pure_o = firrtl.instance pure @Pure(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
  // CHECK:     firrtl.matchingconnect %pure_i, %c0_ui1 : !firrtl.uint<1>
  // CHECK:     "unknown"(%pure_o) : (!firrtl.uint<1>) -> ()
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %pure_i, %pure_o = firrtl.instance pure @Pure(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    firrtl.matchingconnect %pure_i, %c0_ui1 : !firrtl.uint<1>
    firrtl.layerblock @A {
      "unknown"(%pure_o) : (!firrtl.uint<1>) -> ()
    }
  }
}

// Effectful instances cannot be moved.
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  firrtl.module @Effectful(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
    firrtl.matchingconnect %o, %i : !firrtl.uint<1>
    "unknown"() : () -> ()
  }
  // CHECK: firrtl.module @Top() {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %effectful_i, %effectful_o = firrtl.instance effectful @Effectful(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
  // CHECK:   firrtl.matchingconnect %effectful_i, %c0_ui1 : !firrtl.uint<1>
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     "unknown"(%effectful_o) : (!firrtl.uint<1>) -> ()
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %effectful_i, %effectful_o = firrtl.instance effectful @Effectful(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    firrtl.matchingconnect %effectful_i, %c0_ui1 : !firrtl.uint<1>
    firrtl.layerblock @A {
      "unknown"(%effectful_o) : (!firrtl.uint<1>) -> ()
    }
  }
}

// Parents of effectful instances cannot be moved.
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  firrtl.module @Effectful(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
    firrtl.matchingconnect %o, %i : !firrtl.uint<1>
    "unknown"() : () -> ()
  }
  firrtl.module @EffectfulParent(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
    %effectful_i, %effectful_o = firrtl.instance effectful @Effectful(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    firrtl.matchingconnect %effectful_i, %i : !firrtl.uint<1>
    firrtl.matchingconnect %o, %effectful_o : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @Top() {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %parent_i, %parent_o = firrtl.instance parent @EffectfulParent(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
  // CHECK:   firrtl.matchingconnect %parent_i, %c0_ui1 : !firrtl.uint<1>
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     "unknown"(%parent_o) : (!firrtl.uint<1>) -> ()
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %parent_i, %parent_o = firrtl.instance parent @EffectfulParent(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    firrtl.matchingconnect %parent_i, %c0_ui1 : !firrtl.uint<1>
    firrtl.layerblock @A {
      "unknown"(%parent_o) : (!firrtl.uint<1>) -> ()
    }
  }
}

// Modules with a layerblock cannot be moved, regardless of effectfulness. This
// would result in a bind-under-bind.
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  firrtl.module @HasLayerblock(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
    firrtl.matchingconnect %o, %i : !firrtl.uint<1>
    firrtl.layerblock @A {}
  }
  // CHECK: firrtl.module @Top() {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %hasLayerblock_i, %hasLayerblock_o = firrtl.instance hasLayerblock @HasLayerblock(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
  // CHECK:   firrtl.matchingconnect %hasLayerblock_i, %c0_ui1 : !firrtl.uint<1>
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     "unknown"(%hasLayerblock_o) : (!firrtl.uint<1>) -> ()
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %hasLayerblock_i, %hasLayerblock_o = firrtl.instance hasLayerblock @HasLayerblock(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    firrtl.matchingconnect %hasLayerblock_i, %c0_ui1 : !firrtl.uint<1>
    firrtl.layerblock @A {
      "unknown"(%hasLayerblock_o) : (!firrtl.uint<1>) -> ()
    }
  }
}

// Parents of modules with a layerblock  cannot be moved.
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  firrtl.module @HasLayerblock(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
    firrtl.matchingconnect %o, %i : !firrtl.uint<1>
    firrtl.layerblock @A {}
  }
  firrtl.module @HasLayerblockParent(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
    %hasLayerblock_i, %hasLayerblock_o = firrtl.instance hasLayerblock @HasLayerblock(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    firrtl.matchingconnect %hasLayerblock_i, %i : !firrtl.uint<1>
    firrtl.matchingconnect %o, %hasLayerblock_o : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @Top() {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %hasLayerblockParent_i, %hasLayerblockParent_o = firrtl.instance hasLayerblockParent @HasLayerblockParent(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
  // CHECK:   firrtl.matchingconnect %hasLayerblockParent_i, %c0_ui1 : !firrtl.uint<1>
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     "unknown"(%hasLayerblockParent_o) : (!firrtl.uint<1>) -> ()
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %hasLayerblockParent_i, %hasLayerblockParent_o = firrtl.instance hasLayerblockParent @HasLayerblockParent(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    firrtl.matchingconnect %hasLayerblockParent_i, %c0_ui1 : !firrtl.uint<1>
    firrtl.layerblock @A {
      "unknown"(%hasLayerblockParent_o) : (!firrtl.uint<1>) -> ()
    }
  }
}

//===----------------------------------------------------------------------===//
// Connects at end / dominance preservation.
//===----------------------------------------------------------------------===//

// Connect gets dragged forwards into layerblock.
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  // CHECK: firrtl.module @Top() {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   "unknown"(%c0_ui1) : (!firrtl.uint<1>) -> ()
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     %w = firrtl.wire : !firrtl.bundle<a: uint<1>, b: uint<1>>
  // CHECK:     %0 = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<1>, b: uint<1>>
  // CHECK:     %1 = firrtl.subfield %w[b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
  // CHECK:     %c0_ui1_0 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:     firrtl.matchingconnect %0, %c0_ui1_0 : !firrtl.uint<1>
  // CHECK:     firrtl.matchingconnect %1, %c0_ui1_0 : !firrtl.uint<1>
  // CHECK:     "unknown"(%1) : (!firrtl.uint<1>) -> ()
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top() {
    %w = firrtl.wire : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %0 = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %1 = firrtl.subfield %w[b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.layerblock @A {
      "unknown"(%1) : (!firrtl.uint<1>) -> ()
    }
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    "unknown"(%c0_ui1) : (!firrtl.uint<1>) -> ()
    firrtl.matchingconnect %0, %c0_ui1 : !firrtl.uint<1>
    firrtl.matchingconnect %1, %c0_ui1 : !firrtl.uint<1>
  }
}

// Check that layer-sink doesn't accidentally un-nest layerblocks while
// rearranging the order of ops.
firrtl.circuit "Top" {
  firrtl.layer @L1 bind {
    firrtl.layer @L2 bind {}
  }
  // CHECK: firrtl.module @Top() {
  // CHECK:   firrtl.layerblock @L1 {
  // CHECK:     firrtl.layerblock @L1::@L2 {
  // CHECK:       %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:       %w = firrtl.wire : !firrtl.uint<1>
  // CHECK:       firrtl.matchingconnect %w, %c0_ui1 : !firrtl.uint<1>
  // CHECK:       "unknown"(%w) : (!firrtl.uint<1>) -> ()
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %w = firrtl.wire : !firrtl.uint<1>
    firrtl.layerblock @L1 {
      firrtl.layerblock @L1::@L2 {
        "unknown"(%w) : (!firrtl.uint<1>) -> ()
      }
    }
    firrtl.matchingconnect %w, %c0_ui1 : !firrtl.uint<1>
  }
}

//===----------------------------------------------------------------------===//
// Connects that are buried in layerblocks, will stay in layerblocks.
//===----------------------------------------------------------------------===//

// Ref-define buried inside layerblock drives a coloured probe port.
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  // CHECK: firrtl.module @Top(out %port: !firrtl.probe<uint<1>, @A>) {
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:     %0 = firrtl.ref.send %c0_ui1 : !firrtl.uint<1>
  // CHECK:     %1 = firrtl.ref.cast %0 : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>, @A>
  // CHECK:     firrtl.ref.define %port, %1 : !firrtl.probe<uint<1>, @A>
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top(out %port: !firrtl.probe<uint<1>, @A>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.ref.send %c0_ui1 : !firrtl.uint<1>
    %1 = firrtl.ref.cast %0 : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>, @A>
    firrtl.layerblock @A {
      firrtl.ref.define %port, %1 : !firrtl.probe<uint<1>, @A>
    }
  }
}

// Ref-define buried inside layerblock drives a coloured wire.
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  // CHECK: firrtl.module @Top() {
  // CHECK:   %w = firrtl.wire : !firrtl.probe<uint<1>, @A>
  // CHECK:   "unknown"(%w) : (!firrtl.probe<uint<1>, @A>) -> ()
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:     %0 = firrtl.ref.send %c0_ui1 : !firrtl.uint<1>
  // CHECK:     %1 = firrtl.ref.cast %0 : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>, @A>
  // CHECK:     firrtl.ref.define %w, %1 : !firrtl.probe<uint<1>, @A>
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.ref.send %c0_ui1 : !firrtl.uint<1>
    %1 = firrtl.ref.cast %0 : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>, @A>
    %w = firrtl.wire : !firrtl.probe<uint<1>, @A>
    "unknown"(%w) : (!firrtl.probe<uint<1>, @A>) -> ()
    firrtl.layerblock @A {
      firrtl.ref.define %w, %1 : !firrtl.probe<uint<1>, @A>
    }
  }
}

//===----------------------------------------------------------------------===//
// Sinking through Whens.
//===----------------------------------------------------------------------===//

// Sink an op that is used under a when, without sinking into the when.
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  // CHECK: firrtl.module @Top(in %port: !firrtl.uint<1>) {
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     %node = firrtl.node %port : !firrtl.uint<1>
  // CHECK:     firrtl.when %port : !firrtl.uint<1> {
  // CHECK:       "unknown"(%node) : (!firrtl.uint<1>) -> ()
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top(in %port: !firrtl.uint<1>) {
    %node = firrtl.node %port : !firrtl.uint<1>
    firrtl.layerblock @A {
      firrtl.when %port : !firrtl.uint<1> {
        "unknown"(%node) : (!firrtl.uint<1>) -> ()
      }
    }
  }
}

// Sink an op that is used under both arms of a when, without sinking into the when.
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  // CHECK: firrtl.module @Top(in %port: !firrtl.uint<1>) {
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     %node = firrtl.node %port : !firrtl.uint<1>
  // CHECK:     firrtl.when %port : !firrtl.uint<1> {
  // CHECK:       "unknown"(%node) : (!firrtl.uint<1>) -> ()
  // CHECK:     } else {
  // CHECK:       "unknown"(%node) : (!firrtl.uint<1>) -> ()
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top(in %port: !firrtl.uint<1>) {
    %node = firrtl.node %port : !firrtl.uint<1>
    firrtl.layerblock @A {
      firrtl.when %port : !firrtl.uint<1> {
        "unknown"(%node) : (!firrtl.uint<1>) -> ()
      } else {
        "unknown"(%node) : (!firrtl.uint<1>) -> ()
      }
    }
  }
}

// Check that sinking occurs within a when.
firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  // CHECK: firrtl.module @Top(in %port: !firrtl.uint<1>) {
  // CHECK:   firrtl.when %port : !firrtl.uint<1> {
  // CHECK:     firrtl.layerblock @A {
  // CHECK:       %node = firrtl.node %port : !firrtl.uint<1>
  // CHECK:       "unknown"(%node) : (!firrtl.uint<1>) -> ()
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top(in %port: !firrtl.uint<1>) {
    firrtl.when %port : !firrtl.uint<1> {
      %node = firrtl.node %port : !firrtl.uint<1>
      firrtl.layerblock @A {
        "unknown"(%node) : (!firrtl.uint<1>) -> ()
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Cloning of constants.
//===----------------------------------------------------------------------===//

firrtl.circuit "Top" {
  firrtl.layer @A bind {}
  firrtl.layer @B bind {}

  // CHECK: firrtl.module @Top() {
  // CHECK-NOT: %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:     "unknown"(%c0_ui1) : (!firrtl.uint<1>) -> ()
  // CHECK:   }
  // CHECK:   firrtl.layerblock @A {
  // CHECK:     %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:     "unknown"(%c0_ui1) : (!firrtl.uint<1>) -> ()
  // CHECK:   }
  // CHECK: }
  firrtl.module @Top() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.layerblock @A {
      "unknown"(%c0_ui1) : (!firrtl.uint<1>) -> ()
    }
    firrtl.layerblock @A {
      "unknown"(%c0_ui1) : (!firrtl.uint<1>) -> ()
    }
  }
}

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//

// Tests of things which do not currently sink, but should.
//
// CHECK-LABEL: firrtl.circuit "LayerSinkExpectedFailures"
firrtl.circuit "LayerSinkExpectedFailures" {
  firrtl.layer @Subaccess bind {}
  firrtl.layer @Subfield bind {}
  firrtl.layer @Subindex bind {}

  // CHECK: firrtl.module @LayerSinkExpectedFailures
  firrtl.module @LayerSinkExpectedFailures(in %a: !firrtl.uint<1>) {
    // CHECK-NEXT: firrtl.layerblock @Subaccess {
    // CHECK-NEXT:   %wire_subaccess = firrtl.wire
    // CHECK-NEXT:   %0 = firrtl.subaccess %wire_subaccess[%a] : !firrtl.vector<uint<1>, 2>, !firrtl.uint<1>
    // CHECK-NEXT:   firrtl.matchingconnect %0, %0 : !firrtl.uint<1>
    // CHECK-NEXT:   "unknown"(%wire_subaccess) : (!firrtl.vector<uint<1>, 2>) -> ()
    // CHECK-NEXT: }
    %wire_subaccess = firrtl.wire : !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subaccess %wire_subaccess[%a] : !firrtl.vector<uint<1>, 2>, !firrtl.uint<1>
    firrtl.matchingconnect %0, %0 : !firrtl.uint<1>
    firrtl.layerblock @Subaccess {
      "unknown"(%wire_subaccess) : (!firrtl.vector<uint<1>, 2>) -> ()
    }

    // CHECK-NEXT: firrtl.layerblock @Subfield {
    // CHECK-NEXT:   %wire_subfield = firrtl.wire
    // CHECK-NEXT:   %0 = firrtl.subfield %wire_subfield[a]
    // CHECK-NEXT:   firrtl.matchingconnect %0, %0
    // CHECK-NEXT:   "unknown"(%wire_subfield) : (!firrtl.bundle<a: uint<1>>) -> ()
    // CHECK-NEXT: }
    %wire_subfield = firrtl.wire : !firrtl.bundle<a: uint<1>>
    %1 = firrtl.subfield %wire_subfield[a] : !firrtl.bundle<a: uint<1>>
    firrtl.matchingconnect %1, %1 : !firrtl.uint<1>
    firrtl.layerblock @Subfield {
      "unknown"(%wire_subfield) : (!firrtl.bundle<a: uint<1>>) -> ()
    }

    // CHECK-NEXT: firrtl.layerblock @Subindex {
    // CHECK-NEXT:   %wire_subindex = firrtl.wire
    // CHECK-NEXT:   %0 = firrtl.subindex %wire_subindex[0]
    // CHECK-NEXT:   firrtl.matchingconnect %0, %0
    // CHECK-NEXT:   "unknown"(%wire_subindex) : (!firrtl.vector<uint<1>, 1>) -> ()
    // CHECK-NEXT: }
    %wire_subindex = firrtl.wire : !firrtl.vector<uint<1>, 1>
    %2 = firrtl.subindex %wire_subindex[0] : !firrtl.vector<uint<1>, 1>
    firrtl.matchingconnect %2, %2 : !firrtl.uint<1>
    firrtl.layerblock @Subindex {
      "unknown"(%wire_subindex) : (!firrtl.vector<uint<1>, 1>) -> ()
    }
  }
}

firrtl.circuit "Sub" {
  firrtl.layer @Subfield bind {}
  firrtl.module @Sub() {
    %w = firrtl.wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.bundle<a: uint<1>>
    %w_a = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<1>>
    %z = firrtl.constant 0 : !firrtl.uint<1>
    // {{op connects to a destination which is defined outside its enclosing layer block}}
    firrtl.matchingconnect %w_a, %z: !firrtl.uint<1>
    firrtl.layerblock @Subfield {
      firrtl.node %w_a : !firrtl.uint<1>
    }
  }
}

// Test that a port annotation on a module prevents us from sinking instances of
// that module into layerblocks.
firrtl.circuit "DoNotSinkInstanceOfModuleWithPortAnno" {
  firrtl.layer @A bind {}
  firrtl.module @ModuleWithPortAnno(out %out : !firrtl.uint<1>)
    attributes {
      portAnnotations = [
        [{class = "circt.FullResetAnnotation", resetType = "async"}]
      ]
    }
  {}

  // CHECK: firrtl.module @DoNotSinkInstanceOfModuleWithPortAnno
  firrtl.module @DoNotSinkInstanceOfModuleWithPortAnno() {
    // CHECK-NEXT: firrtl.instance foo @ModuleWithPortAnn
    %foo_out = firrtl.instance foo @ModuleWithPortAnno(out out : !firrtl.uint<1>)
    // CHECK-NEXT: firrtl.layerblock 
    firrtl.layerblock @A {
      "unknown"(%foo_out) : (!firrtl.uint<1>) -> ()
    }
  }
}

// CHECK-LABEL: firrtl.circuit "SinkXMRs"
firrtl.circuit "SinkXMRs" {
  firrtl.layer @A bind {}

  hw.hierpath @xmr [@SinkXMRs::@target]

  firrtl.module public @SinkXMRs() {
    %target = firrtl.wire sym @target : !firrtl.uint<1>
    %0 = firrtl.xmr.deref @xmr : !firrtl.uint<1>
    %1 = firrtl.xmr.ref @xmr : !firrtl.ref<uint<1>>
  
    // CHECK: firrtl.layerblock @A
    firrtl.layerblock @A {
      // CHECK-NEXT: %0 = firrtl.xmr.deref @xmr : !firrtl.uint<1>
      // CHECK-NEXT: %1 = firrtl.xmr.ref @xmr : !firrtl.probe<uint<1>>
      "unknown"(%0, %1) : (!firrtl.uint<1>, !firrtl.ref<uint<1>>) -> ()
    }
  }
}
