// RUN: circt-opt --verify-diagnostics --verify-roundtrip %s | FileCheck %s

module {
// CHECK-LABEL: moore.class.classdecl @Plain {
// CHECK: }
moore.class.classdecl @Plain {
}

// CHECK-LABEL:   moore.class.classdecl @I {
// CHECK:   }
moore.class.classdecl @I {
}

// CHECK-LABEL:   moore.class.classdecl @Base {
// CHECK:   }
// CHECK:   moore.class.classdecl @Derived extends @Base {
// CHECK:   }
moore.class.classdecl @Base {
}
moore.class.classdecl @Derived extends @Base {
}

// CHECK-LABEL:   moore.class.classdecl @IBase {
// CHECK:   }
// CHECK:   moore.class.classdecl @IExt extends @IBase {
// CHECK:   }

moore.class.classdecl @IBase {
}
moore.class.classdecl @IExt extends @IBase {
}

// CHECK-LABEL:   moore.class.classdecl @IU {
// CHECK:   }
// CHECK:   moore.class.classdecl @C1 implements [@IU] {
// CHECK:   }
moore.class.classdecl @IU {
}
moore.class.classdecl @C1 implements [@IU] {
}

// CHECK-LABEL:   moore.class.classdecl @I1 {
// CHECK:   }
// CHECK:   moore.class.classdecl @I2 {
// CHECK:   }
// CHECK:   moore.class.classdecl @C2 implements [@I1, @I2] {
// CHECK:   }
moore.class.classdecl @I1 {
}
moore.class.classdecl @I2 {
}
moore.class.classdecl @C2 implements [@I1, @I2] {
}

// CHECK-LABEL:   moore.class.classdecl @B {
// CHECK:   }
// CHECK:   moore.class.classdecl @J1 {
// CHECK:   }
// CHECK:   moore.class.classdecl @J2 {
// CHECK:   }
// CHECK:   moore.class.classdecl @D extends @B implements [@J1, @J2] {
// CHECK:   }
moore.class.classdecl @B {
}
moore.class.classdecl @J1 {
}
moore.class.classdecl @J2 {
}
moore.class.classdecl @D extends @B implements [@J1, @J2] {
}

// CHECK-LABEL:   moore.class.classdecl @PropertyCombo {
// CHECK-NEXT:     moore.class.propertydecl @pubAutoI32 : !moore.i32
// CHECK-NEXT:     moore.class.propertydecl @protStatL18 : !moore.l18
// CHECK-NEXT:     moore.class.propertydecl @localAutoI32 : !moore.i32
// CHECK:   }
moore.class.classdecl @PropertyCombo {
  moore.class.propertydecl @pubAutoI32 : !moore.i32
  moore.class.propertydecl @protStatL18 : !moore.l18
  moore.class.propertydecl @localAutoI32 : !moore.i32
}


}
