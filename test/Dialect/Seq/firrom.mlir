// RUN: circt-opt %s --verify-diagnostics | circt-opt --verify-diagnostics | FileCheck %s

// CHECK-LABEL: hw.module @Basic
hw.module @Basic() {
  // CHECK-NEXT: seq.firrom 0 : <3 x 19>
  %0 = seq.firrom 0 : <3 x 19>

  // CHECK-NEXT: seq.firrom sym @someMem 0 : <3 x 19>
  %1 = seq.firrom sym @someMem 0 : <3 x 19>

  // CHECK-NEXT: %myMem1 = seq.firrom 0 : <3 x 19>
  %2 = seq.firrom name "myMem1" 0 : <3 x 19>

  // CHECK-NEXT: %myMem2 = seq.firrom 0 : <3 x 19>
  %myMem2 = seq.firrom 0 : <3 x 19>

  // CHECK-NEXT: %myMem3 = seq.firrom 0 : <3 x 19>
  %ignoredName = seq.firrom name "myMem3" 0 : <3 x 19>

  // CHECK-NEXT: seq.firrom 0 {init = #seq.firrom.init<"mem.txt", false, false>} : <3 x 19>
  %3 = seq.firrom 0 {init = #seq.firrom.init<"mem.txt", false, false>} : <3 x 19>
}

// CHECK-LABEL: hw.module @Ports
hw.module @Ports(%clock: i1, %enable: i1, %address: i4, %data: i20, %mode: i1, %mask: i4) {
  // CHECK-NEXT: %mem = seq.firrom 0 : <12 x 20>
  %mem = seq.firrom 0 : <12 x 20>

  // Read ports
  // CHECK-NEXT: [[R0:%.+]] = seq.firrom.read_port %mem[%address], clock %clock : <12 x 20>
  // CHECK-NEXT: [[R1:%.+]] = seq.firrom.read_port %mem[%address], clock %clock enable %enable : <12 x 20>
  %0 = seq.firrom.read_port %mem[%address], clock %clock : <12 x 20>
  %1 = seq.firrom.read_port %mem[%address], clock %clock enable %enable : <12 x 20>

  // CHECK-NEXT: comb.xor [[R0]], [[R1]]
  comb.xor %0, %1: i20
}
