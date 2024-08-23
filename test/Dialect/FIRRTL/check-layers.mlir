// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-layers))' %s | FileCheck %s

// CHECK-LABEL: firrtl.circuit "CheckLayers"
firrtl.circuit "CheckLayers" {
  firrtl.layer @A bind {}
  firrtl.module @CheckLayers() {
    firrtl.layerblock @A {
      firrtl.instance nolayers @NoLayers()
    }
    firrtl.instance layers @Layers()
  }
  firrtl.module @NoLayers() { }
  firrtl.module @Layers() {
    firrtl.layerblock @A {}
  }
}
