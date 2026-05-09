// RUN: circt-opt %s --verify-diagnostics --split-input-file

func.func @Foo0(%state: !arc.state<i8>, %val: i8) {
  // expected-error @below {{`@Empty0` does not reference a valid `arc.model`}}
  arc.state_write %state = %val tap @Empty0[1] : <i8>
  return
}

// -----

hw.module @Empty1() {}

func.func @Foo1(%state: !arc.state<i8>, %val: i8) {
  // expected-error @below {{`@Empty1` does not reference a valid `arc.model`}}
  arc.state_write %state = %val tap @Empty1[1] : <i8>
  return
}

// -----

arc.model @Empty2 io !hw.modty<>  {
  ^bb0(%arg0: !arc.storage):
}

func.func @Foo2(%state: !arc.state<i8>, %val: i8) {
  // expected-error @below {{referenced model has no trace metadata}}
  arc.state_write %state = %val tap @Empty2[1] : <i8>
  return
}

// -----

arc.model @Empty3 io !hw.modty<>  traceTaps [#arc.trace_tap<i8, 0, ["foo"]>] {
  ^bb0(%arg0: !arc.storage):
}

func.func @Foo3(%state: !arc.state<i8>, %val: i8) {
  // expected-error @below {{tap index exceeds model's tap array}}
  arc.state_write %state = %val tap @Empty3[1] : <i8>
  return
}

// -----

arc.model @Empty4 io !hw.modty<> traceTaps [#arc.trace_tap<i8, 0, ["foo"]>, #arc.trace_tap<i1, 1, ["bar"]>] {
  ^bb0(%arg0: !arc.storage):
}

func.func @Foo4(%state: !arc.state<i8>, %val: i8) {
  // expected-error @below {{incorrect signal type in referenced tap attribute}}
  arc.state_write %state = %val tap @Empty4[1] : <i8>
  return
}
