// A diagnosed debug intrinsic must fail the pass, not merely print. When the
// converter reports an error but still returns success, the tool exits 0 and
// leaves the unlowered `firrtl.int.generic` in the IR, so every downstream pass
// sees an intrinsic that was already reported as broken.
//
// RUN: not circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-lower-intrinsics)))' %s --split-input-file 2>&1 | FileCheck %s

// CHECK: error: debug enum: failed to parse 'variants'
// CHECK: failed to legalize operation 'firrtl.int.generic'
firrtl.circuit "EnumBadJSON" {
  firrtl.module @EnumBadJSON(in %s: !firrtl.uint<2>) {
    firrtl.int.generic "circt_debug_var"
      <name: none = "s", typeName: none = "MyState", enumFqn: none = "pkg.MyState$",
       variants: none = "[{not valid json">
      %s : (!firrtl.uint<2>) -> ()
  }
}

// -----

// CHECK: error: circt_debug_var: name 'x' is ambiguous
// CHECK: failed to legalize operation 'firrtl.int.generic'
firrtl.circuit "AmbiguousName" {
  firrtl.module @AmbiguousName() {
    %x = firrtl.wire : !firrtl.uint<8>
    %x_mem = chirrtl.combmem {name = "x"} : !chirrtl.cmemory<uint<8>, 4>
    firrtl.int.generic "circt_debug_var"
      <name: none = "x", typeName: none = "UInt"> : () -> ()
  }
}
