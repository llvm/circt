// RUN: circt-opt %s | FileCheck %s

firrtl.circuit "Classes" {
  firrtl.module @Classes() {}

  // CHECK-LABEL: firrtl.class @StringOut(out %str: !firrtl.string)
  firrtl.class @StringOut(out %str: !firrtl.string) {
    %0 = firrtl.string "foo"
    firrtl.propassign %str, %0 : !firrtl.string
  }

  // CHECK-LABEL: firrtl.class @StringPassThru(in %in_str: !firrtl.string, out %out_str: !firrtl.string)
  firrtl.class @StringPassThru(in %in_str: !firrtl.string, out %out_str: !firrtl.string) {
    // CHECK: firrtl.propassign %out_str, %in_str : !firrtl.string
    firrtl.propassign %out_str, %in_str : !firrtl.string
  }

  // CHECK-LABEL: firrtl.module @ModuleWithObjectPort(in %in: !firrtl.class<@StringPassThru(in in_str: !firrtl.string, out out_str: !firrtl.string)>)
  firrtl.module @ModuleWithObjectPort(in %in: !firrtl.class<@StringPassThru(in in_str: !firrtl.string, out out_str: !firrtl.string)>) {}

  // CHECK-LABEL: firrtl.class @EmptyClass()
  firrtl.class @EmptyClass() {}

  // CHECK-LABEL: firrtl.module @ModuleWithOutputObject(out %out: !firrtl.class<@EmptyClass()>)
  firrtl.module @ModuleWithOutputObject(out %out: !firrtl.class<@EmptyClass()>) {
    %0 = firrtl.object @EmptyClass()
    firrtl.propassign %out, %0 : !firrtl.class<@EmptyClass()>
  }

  // CHECK-LABEL: firrtl.module @ObjectSubfieldOp(out %out: !firrtl.string)
  firrtl.module @ObjectSubfieldOp(out %out: !firrtl.string) {
    // CHECK: %0 = firrtl.object @StringOut(out str: !firrtl.string)
    // CHECK: %1 = firrtl.object.subfield %0[str] : !firrtl.class<@StringOut(out str: !firrtl.string)>
    // CHECK: firrtl.propassign %out, %1 : !firrtl.string
    %0 = firrtl.object @StringOut(out str: !firrtl.string)
    %1 = firrtl.object.subfield %0[str] : !firrtl.class<@StringOut(out str: !firrtl.string)>
    firrtl.propassign %out, %1 : !firrtl.string
  }

  // CHECK-LABEL: firrtl.module @PropassignObjectPort(out %out: !firrtl.string)
  firrtl.module @PropassignObjectPort(out %out: !firrtl.string) {
    // CHECK: %0 = firrtl.object @StringPassThru(in in_str: !firrtl.string, out out_str: !firrtl.string)
    // CHECK: %1 = firrtl.string "foo"
    // CHECK: %2  = firrtl.object.subfield %0[in_str] : !firrtl.class<@StringPassThru(in in_str: !firrtl.string, out out_str: !firrtl.string)>
    // CHECK: %3 = firrtl.object.subfield %0[out_str] : !firrtl.class<@StringPassThru(in in_str: !firrtl.string, out out_str: !firrtl.string)>
    // CHECK: firrtl.propassign %2, %1 : !firrtl.string
    // CHECK: firrtl.propassign %out, %3 : !firrtl.string
    %0 = firrtl.object @StringPassThru(in in_str: !firrtl.string, out out_str: !firrtl.string)
    %1 = firrtl.string "foo"
    %2  = firrtl.object.subfield %0[in_str] : !firrtl.class<@StringPassThru(in in_str: !firrtl.string, out out_str: !firrtl.string)>
    %3 = firrtl.object.subfield %0[out_str] : !firrtl.class<@StringPassThru(in in_str: !firrtl.string, out out_str: !firrtl.string)>
    firrtl.propassign %2, %1 : !firrtl.string
    firrtl.propassign %out, %3 : !firrtl.string
  }

  firrtl.module @ModuleWithInputString(in %in : !firrtl.string) {}

    // CHECK-LABEL: firrtl.module @PropassignToInstancePort(in %in: !firrtl.string)
  firrtl.module @PropassignToInstancePort(in %in: !firrtl.string) {
    // CHECK: %foo_in = firrtl.instance foo @ModuleWithInputString(in in: !firrtl.string)
    // CHECK: firrtl.propassign %foo_in, %in : !firrtl.string
    %foo_in = firrtl.instance foo @ModuleWithInputString(in in: !firrtl.string)
    firrtl.propassign %foo_in, %in : !firrtl.string
  }

  firrtl.module @ModuleWithOutputString(out %out : !firrtl.string) {
    %0 = firrtl.string "foo"
    firrtl.propassign %out, %0 : !firrtl.string
  }

  // CHECK-LABEL: firrtl.module @PropassignFromInstancePort(out %out: !firrtl.string)
  firrtl.module @PropassignFromInstancePort(out %out: !firrtl.string) {
    // CHECK: %foo_out = firrtl.instance foo @ModuleWithOutputString(out out: !firrtl.string)
    // CHECK: firrtl.propassign %out, %foo_out : !firrtl.string
    %foo_out = firrtl.instance foo @ModuleWithOutputString(out out: !firrtl.string)
    firrtl.propassign %out, %foo_out : !firrtl.string
  }

  // CHECK-LABEL: firrtl.extclass @MyExtClass(out val: !firrtl.string)
  firrtl.extclass @MyExtClass(out val: !firrtl.string)

  // CHECK-LABEL: firrtl.module @UserOfExtClass(out %port: !firrtl.class<@MyExtClass(out val: !firrtl.string)>)
  firrtl.module @UserOfExtClass(out %port: !firrtl.class<@MyExtClass(out val: !firrtl.string)>) {
    // CHECK: %0 = firrtl.object @MyExtClass(out val: !firrtl.string)
    // CHECK: firrtl.propassign %port, %0 : !firrtl.class<@MyExtClass(out val: !firrtl.string)>
    %0 = firrtl.object @MyExtClass(out val: !firrtl.string)
    firrtl.propassign %port, %0 : !firrtl.class<@MyExtClass(out val: !firrtl.string)>
  }

}
