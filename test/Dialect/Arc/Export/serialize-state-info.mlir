// RUN: circt-translate %s --export-arc-model-info | FileCheck %s

// CHECK-LABEL: "name": "Foo"
// CHECK-DAG: "numStateBytes": 5724
arc.model @Foo io !hw.modty<input a : i19, output b : i42> storageBytes 5724 {
^bb0(%arg0: !arc.storage):
  // CHECK:      "name": "a"
  // CHECK-NEXT: "offset": 0
  // CHECK-NEXT: "numBits": 19
  // CHECK-NEXT: "type": "input"
  arc.root_input "a", %arg0 {offset = 0} : (!arc.storage) -> !arc.state<i19>

  // CHECK:      "name": "b"
  // CHECK-NEXT: "offset": 16
  // CHECK-NEXT: "numBits": 42
  // CHECK-NEXT: "type": "output"
  arc.root_output "b", %arg0 {offset = 16} : (!arc.storage) -> !arc.state<i42>
}

// CHECK-LABEL: "name": "Bar"
// CHECK-DAG: "numStateBytes": 9001
arc.model @Bar io !hw.modty<> storageBytes 9001  {
^bb0(%arg0: !arc.storage):
  // CHECK-NOT: "offset": "420"
  arc.alloc_state %arg0 {offset = 420} : (!arc.storage) -> !arc.state<i11>

  // CHECK:      "name": "x"
  // CHECK-NEXT: "offset": 24
  // CHECK-NEXT: "numBits": 63
  // CHECK-NEXT: "type": "register"
  arc.alloc_state %arg0 {name = "x", offset = 24} : (!arc.storage) -> !arc.state<i63>

  // CHECK:      "name": "y"
  // CHECK-NEXT: "offset": 48
  // CHECK-NEXT: "numBits": 17
  // CHECK-NEXT: "type": "memory"
  // CHECK-NEXT: "stride": 3
  // CHECK-NEXT: "depth": 5
  arc.alloc_memory %arg0 {name = "y", offset = 48, stride = 3} : (!arc.storage) -> !arc.memory<5 x i17, i3>

  // CHECK:      "name": "z"
  // CHECK-NEXT: "offset": 92
  // CHECK-NEXT: "numBits": 1337
  // CHECK-NEXT: "type": "wire"
  arc.alloc_state %arg0 tap {name = "z", offset = 92} : (!arc.storage) -> !arc.state<i1337>
}

// CHECK-LABEL: "name": "Alpha"
// CHECK: "initialFnSym": "AlphaInitialize"
// CHECK: "finalFnSym": "AlphaFinalize"
arc.model @Alpha io !hw.modty<> storageBytes 4 initializer @AlphaInitialize finalizer @AlphaFinalize {
^bb0(%arg0: !arc.storage):
}

func.func private @AlphaInitialize(!arc.storage)
func.func private @AlphaFinalize(!arc.storage)
