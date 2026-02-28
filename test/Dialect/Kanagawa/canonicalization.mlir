// RUN: circt-opt --allow-unregistered-dialect --split-input-file --canonicalize --cse %s | FileCheck %s

kanagawa.design @foo {

kanagawa.container sym @C {
}

// CHECK-LABEL:   kanagawa.container sym @AccessChild {
// CHECK:           %[[VAL_1:.*]] = kanagawa.container.instance @c, <@foo::@C
// CHECK:           "foo.user"(%[[VAL_1]]) : (!kanagawa.scoperef<@foo::@C>) -> ()
// CHECK:         }
kanagawa.container sym @AccessChild {
  %c = kanagawa.container.instance @c, <@foo::@C>
  %c_ref = kanagawa.path [
    #kanagawa.step<child , @c : !kanagawa.scoperef<@foo::@C>>
  ]
  "foo.user"(%c_ref) : (!kanagawa.scoperef<@foo::@C>) -> ()
}
}
