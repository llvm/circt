// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce --test /usr/bin/env --test-arg true --include firrtl-module-externalizer --keep-best=0 %s | FileCheck %s

// CHECK-LABEL: firrtl.circuit "Externalize"
firrtl.circuit "Externalize" {
  // CHECK: firrtl.extmodule @Externalize()
  firrtl.module @Externalize() {}

  // CHECK: firrtl.module @SkipInnerRefs()
  firrtl.module @SkipInnerRefs() {
    firrtl.instance inst sym @sym {doNotPrint} @Externalize()
  }

  firrtl.bind <@SkipInnerRefs::@sym>
}
