// RUN: circt-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: @connect_ports
// CHECK-SAME: (inout %[[IN:.+]] : [[TYPE:.+]], inout %[[OUT:.+]] : [[TYPE]])
// CHECK-NEXT: llhd.con %[[IN]], %[[OUT]] : !hw.inout<[[TYPE]]>
hw.module @connect_ports(inout %in: i32, inout %out: i32) {
  llhd.con %in, %out : !hw.inout<i32>
}
