// RUN: circt-opt %s --arc-insert-runtime | FileCheck %s

// The models map is iterated to create the per-model runtime ops, so its
// iteration order reaches emitted op order. The arc.runtime.model ops must
// follow module op order, not pointer-keyed map iteration order.

// CHECK: arc.runtime.model @arcRuntimeModel_zebra "zebra"
// CHECK: arc.runtime.model @arcRuntimeModel_alpha "alpha"
// CHECK: arc.runtime.model @arcRuntimeModel_quebec "quebec"
// CHECK: arc.runtime.model @arcRuntimeModel_bravo "bravo"

arc.model @zebra io !hw.modty<input clk : i1, output o : i8> storageBytes 4 {
^bb0(%arg0: !arc.storage):
  %0 = arc.root_input "clk", %arg0 {offset = 0 : i32} : (!arc.storage) -> !arc.state<i1>
  %1 = arc.alloc_state %arg0 {offset = 1 : i32} : (!arc.storage) -> !arc.state<i8>
  %out_o = arc.root_output "o", %arg0 {offset = 3 : i32} : (!arc.storage) -> !arc.state<i8>
  %2 = arc.state_read %1 : <i8>
  arc.state_write %out_o = %2 : <i8>
}
arc.model @alpha io !hw.modty<input clk : i1, output o : i8> storageBytes 4 {
^bb0(%arg0: !arc.storage):
  %0 = arc.root_input "clk", %arg0 {offset = 0 : i32} : (!arc.storage) -> !arc.state<i1>
  %1 = arc.alloc_state %arg0 {offset = 1 : i32} : (!arc.storage) -> !arc.state<i8>
  %out_o = arc.root_output "o", %arg0 {offset = 3 : i32} : (!arc.storage) -> !arc.state<i8>
  %2 = arc.state_read %1 : <i8>
  arc.state_write %out_o = %2 : <i8>
}
arc.model @quebec io !hw.modty<input clk : i1, output o : i8> storageBytes 4 {
^bb0(%arg0: !arc.storage):
  %0 = arc.root_input "clk", %arg0 {offset = 0 : i32} : (!arc.storage) -> !arc.state<i1>
  %1 = arc.alloc_state %arg0 {offset = 1 : i32} : (!arc.storage) -> !arc.state<i8>
  %out_o = arc.root_output "o", %arg0 {offset = 3 : i32} : (!arc.storage) -> !arc.state<i8>
  %2 = arc.state_read %1 : <i8>
  arc.state_write %out_o = %2 : <i8>
}
arc.model @bravo io !hw.modty<input clk : i1, output o : i8> storageBytes 4 {
^bb0(%arg0: !arc.storage):
  %0 = arc.root_input "clk", %arg0 {offset = 0 : i32} : (!arc.storage) -> !arc.state<i1>
  %1 = arc.alloc_state %arg0 {offset = 1 : i32} : (!arc.storage) -> !arc.state<i8>
  %out_o = arc.root_output "o", %arg0 {offset = 3 : i32} : (!arc.storage) -> !arc.state<i8>
  %2 = arc.state_read %1 : <i8>
  arc.state_write %out_o = %2 : <i8>
}
