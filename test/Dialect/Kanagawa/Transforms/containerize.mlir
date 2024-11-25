// RUN: circt-opt --pass-pipeline='builtin.module(kanagawa.design(kanagawa-containerize))' %s | FileCheck %s

kanagawa.design @foo {

// CHECK-LABEL:   kanagawa.container sym @A_B
// CHECK-LABEL:   kanagawa.container "MyClassName" sym @MyClass
// CHECK-LABEL:   kanagawa.container sym @A_B_0
// CHECK-LABEL:   kanagawa.container sym @A_C
// CHECK-LABEL:   kanagawa.container sym @A {
// CHECK:           %[[VAL_0:.*]] = kanagawa.this <@foo::@A>
// CHECK:           kanagawa.port.input "A_in" sym @A_in : i1
// CHECK:           %[[VAL_1:.*]] = kanagawa.container.instance @myClass, <@foo::@MyClass>
// CHECK:           %[[VAL_2:.*]] = kanagawa.container.instance @A_B_0, <@foo::@A_B_0>
// CHECK:           %[[VAL_3:.*]] = kanagawa.container.instance @A_C, <@foo::@A_C>

// This container will alias with the @B inside @A, and thus checks the
// name uniquing logic.
kanagawa.container sym @A_B {
  %this = kanagawa.this <@foo::@A_B>
}

kanagawa.class "MyClassName" sym @MyClass {
  %this = kanagawa.this <@foo::@MyClass>
}

kanagawa.class sym @A {
  %this = kanagawa.this <@foo::@A>
  kanagawa.port.input "A_in" sym @A_in : i1
  %myClass = kanagawa.instance @myClass, <@foo::@MyClass>
  kanagawa.container sym @B {
    %B_this = kanagawa.this <@foo::@B>
  }
  kanagawa.container sym @C {
    %C_this = kanagawa.this <@foo::@C>
  }
}

}
