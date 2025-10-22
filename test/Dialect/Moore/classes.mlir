// RUN: circt-opt --verify-diagnostics --verify-roundtrip %s | FileCheck %s

module {
// CHECK-LABEL: moore.class.classdecl  Concrete @Plain : {
// CHECK: }
moore.class.classdecl  Concrete @Plain : {
}

// CHECK-LABEL:   moore.class.classdecl  Interface @I : {
// CHECK:   }
moore.class.classdecl  Interface @I : {
}

// CHECK-LABEL:   moore.class.classdecl  Concrete @Base : {
// CHECK:   }
// CHECK:   moore.class.classdecl  Concrete @Derived extends @Base : {
// CHECK:   }
moore.class.classdecl  Concrete @Base : {
}
moore.class.classdecl  Concrete @Derived extends @Base : {
}

// CHECK-LABEL:   moore.class.classdecl  Interface @IBase : {
// CHECK:   }
// CHECK:   moore.class.classdecl  Interface @IExt extends @IBase : {
// CHECK:   }

moore.class.classdecl  Interface @IBase : {
}
moore.class.classdecl  Interface @IExt extends @IBase : {
}

// CHECK-LABEL:   moore.class.classdecl  Interface @IU : {
// CHECK:   }
// CHECK:   moore.class.classdecl  Concrete @C1 implements [@IU] : {
// CHECK:   }
moore.class.classdecl  Interface @IU : {
}
moore.class.classdecl  Concrete @C1 implements [@IU] : {
}

// CHECK-LABEL:   moore.class.classdecl  Interface @I1 : {
// CHECK:   }
// CHECK:   moore.class.classdecl  Interface @I2 : {
// CHECK:   }
// CHECK:   moore.class.classdecl  Concrete @C2 implements [@I1, @I2] : {
// CHECK:   }
moore.class.classdecl  Interface @I1 : {
}
moore.class.classdecl  Interface @I2 : {
}
moore.class.classdecl  Concrete @C2 implements [@I1, @I2] : {
}

// CHECK-LABEL:   moore.class.classdecl  Concrete @B : {
// CHECK:   }
// CHECK:   moore.class.classdecl  Interface @J1 : {
// CHECK:   }
// CHECK:   moore.class.classdecl  Interface @J2 : {
// CHECK:   }
// CHECK:   moore.class.classdecl  Concrete @D extends @B implements [@J1, @J2] : {
// CHECK:   }
moore.class.classdecl  Concrete @B : {
}
moore.class.classdecl  Interface @J1 : {
}
moore.class.classdecl  Interface @J2 : {
}
moore.class.classdecl  Concrete @D extends @B implements [@J1, @J2] : {
}

// CHECK-LABEL:   moore.class.classdecl  Concrete @PropertyCombo : {
// CHECK-NEXT:     moore.class.propertydecl[ Public,  Automatic] @pubAutoI32 : !moore.i32
// CHECK-NEXT:     moore.class.propertydecl[ Protected,  Static] @protStatL18 : !moore.l18
// CHECK-NEXT:     moore.class.propertydecl[ Local,  Automatic] @localAutoI32 : !moore.i32
// CHECK:   }
moore.class.classdecl  Concrete @PropertyCombo : {
  moore.class.propertydecl[ Public,  Automatic] @pubAutoI32 : !moore.i32
  moore.class.propertydecl[ Protected,  Static] @protStatL18 : !moore.l18
  moore.class.propertydecl[ Local,  Automatic] @localAutoI32 : !moore.i32
}

}
