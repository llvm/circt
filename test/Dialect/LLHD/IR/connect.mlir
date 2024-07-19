// RUN: circt-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: @connect_ports
// CHECK-SAME: (%[[IN:.+]] : [[TYPE:.+]]) ->
// CHECK-SAME: (%[[OUT:.+]] : [[TYPE]])
// CHECK-NEXT: llhd.con %[[IN]], %[[OUT]] : [[TYPE]]
llhd.entity @connect_ports(%in: !hw.inout<i32>) -> (%out: !hw.inout<i32>) {
  llhd.con %in, %out : !hw.inout<i32>
}
