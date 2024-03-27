// RUN: circt-translate %s --export-arc-model-info --split-input-file --verify-diagnostics

arc.model @Foo io !hw.modty<> {
^bb0(%arg0: !arc.storage<42>):
  // expected-error @below {{'arc.alloc_storage' op without allocated offset}}
  arc.alloc_storage %arg0 : (!arc.storage<42>) -> !arc.storage<42>
}

// -----
arc.model @Foo io !hw.modty<> {
^bb0(%arg0: !arc.storage<42>):
  // ignore unnamed
  arc.alloc_state %arg0 : (!arc.storage<42>) -> !arc.state<i1>
  // expected-error @below {{'arc.alloc_state' op without allocated offset}}
  arc.alloc_state %arg0 {name = "foo"} : (!arc.storage<42>) -> !arc.state<i1>
}

// -----
arc.model @Foo io !hw.modty<input foo : i1> {
^bb0(%arg0: !arc.storage<42>):
  // ignore unnamed
  arc.root_input "", %arg0 : (!arc.storage<42>) -> !arc.state<i1>
  // expected-error @below {{'arc.root_input' op without allocated offset}}
  arc.root_input "foo", %arg0 : (!arc.storage<42>) -> !arc.state<i1>
}

// -----
arc.model @Foo io !hw.modty<output foo : i1> {
^bb0(%arg0: !arc.storage<42>):
  // ignore unnamed
  arc.root_output "", %arg0 : (!arc.storage<42>) -> !arc.state<i1>
  // expected-error @below {{'arc.root_output' op without allocated offset}}
  arc.root_output "foo", %arg0 : (!arc.storage<42>) -> !arc.state<i1>
}

// -----
arc.model @Foo io !hw.modty<> {
^bb0(%arg0: !arc.storage<42>):
  // ignore unnamed
  arc.alloc_memory %arg0 : (!arc.storage<42>) -> !arc.memory<4 x i1, i2>
  // expected-error @below {{'arc.alloc_memory' op without allocated offset}}
  arc.alloc_memory %arg0 {name = "foo"} : (!arc.storage<42>) -> !arc.memory<4 x i1, i2>
}

// -----
arc.model @Foo io !hw.modty<> {
^bb0(%arg0: !arc.storage<42>):
  // expected-error @below {{'arc.alloc_memory' op without allocated stride}}
  arc.alloc_memory %arg0 {name = "foo", offset = 8} : (!arc.storage<42>) -> !arc.memory<4 x i1, i2>
}
