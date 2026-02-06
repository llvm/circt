// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --keep-best=0 --include extmodule-instance-remover | FileCheck %s

firrtl.circuit "StringTypes" {
  firrtl.extmodule @Bar(out a: !firrtl.string)
  // CHECK-LABEL: firrtl.module @StringTypes
  firrtl.module @StringTypes() {
    // CHECK-NOT: firrtl.instance Bar @Bar
    %a = firrtl.instance Bar @Bar(out a: !firrtl.string)
    %wire = firrtl.wire : !firrtl.string
    // This should not crash when extmodule-instance-remover handles string types
    firrtl.propassign %wire, %a : !firrtl.string
  }
}

firrtl.circuit "RegularTypes" {
  firrtl.extmodule @Ext(out a: !firrtl.uint<8>)
  // CHECK-LABEL: firrtl.module @RegularTypes
  firrtl.module @RegularTypes() {
    // CHECK-NOT: firrtl.instance Ext @Ext
    // CHECK: %Ext_a = firrtl.wire
    // CHECK: %invalid_ui8 = firrtl.invalidvalue : !firrtl.uint<8>
    // CHECK: firrtl.connect %Ext_a, %invalid_ui8
    %a = firrtl.instance Ext @Ext(out a: !firrtl.uint<8>)
    %wire = firrtl.wire : !firrtl.uint<8>
    firrtl.connect %wire, %a : !firrtl.uint<8>, !firrtl.uint<8>
  }
}
