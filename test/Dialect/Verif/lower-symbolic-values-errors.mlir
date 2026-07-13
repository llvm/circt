// RUN: split-file %s %t
// RUN: circt-opt --verif-lower-symbolic-values=mode=extmodule --verify-diagnostics %t/extmodule.mlir
// RUN: circt-opt --verif-lower-symbolic-values=mode=hw-input --verify-diagnostics %t/hw-input-instantiated.mlir
// RUN: circt-opt --verif-lower-symbolic-values=mode=hw-input --verify-diagnostics %t/hw-input-formal.mlir

//--- extmodule.mlir

hw.module @Foo() {
  // expected-error @below {{symbolic value bit width unknown}}
  verif.symbolic_value : !hw.string
}

//--- hw-input-instantiated.mlir

hw.module @Top(out y : i8) {
  %0 = hw.instance "m" @Instantiated() -> (y: i8)
  hw.output %0 : i8
}

// expected-error @+1 {{cannot lower symbolic values in instantiated module 'Instantiated' to HW inputs; run the 'hw-input' lowering strategy after flattening modules}}
hw.module @Instantiated(out y : i8) {
  %0 = verif.symbolic_value : i8
  hw.output %0 : i8
}

//--- hw-input-formal.mlir

verif.formal @Formal {} {
  // expected-error @below {{cannot lower symbolic value to hw.module input outside of an hw.module}}
  verif.symbolic_value : i8
}
