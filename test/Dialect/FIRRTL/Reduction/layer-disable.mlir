// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce --test /usr/bin/env --test-arg grep --test-arg keep_0 --include firrtl-layer-disable %s | FileCheck %s --implicit-check-not='@A' --implicit-check-not='@B'
// RUN: circt-reduce --test /usr/bin/env --test-arg grep --test-arg keep_1 --include firrtl-layer-disable %s | FileCheck %s --check-prefixes=CHECK,A-LAYER --implicit-check-not='@B'
// RUN: circt-reduce --test /usr/bin/env --test-arg grep --test-arg keep_2 --include firrtl-layer-disable %s | FileCheck %s --check-prefixes=CHECK,A-LAYER,B-LAYER

// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" {
  // A-LAYER-NEXT: firrtl.layer @A
  firrtl.layer @A bind {
    // B-LAYER-NEXT: firrtl.layer @B
    firrtl.layer @B bind {
    }
  }
  firrtl.module @Foo() {
    // CHECK: %keep_0
    %keep_0 = firrtl.wire : !firrtl.uint<1>
    // LAYER-A-NEXT: firrtl.layerblock @A
    firrtl.layerblock @A {
      // LAYER-A-NEXT: %keep_1
      %keep_1 = firrtl.wire : !firrtl.uint<1>
      // LAYER-B-NEXT: firrtl.layerlbock @A::@B
      firrtl.layerblock @A::@B {
        // LAYER-B-NEXT; %keep_2
        %keep_2 = firrtl.wire : !firrtl.uint<1>
      }
    }
  }
}
