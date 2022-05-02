// RUN: circt-opt -hw-generate-case %s | FileCheck %s
// CHECK-LABEL: hw.module @TableLookup(%t_0: i5, %t_1: i5, %t_2: i5, %t_3: i5, %t_4: i5, %t_5: i5, %default: i5, %key: i5) -> (v: i5) {
hw.module @TableLookup(%t_0: i5, %t_1: i5, %t_2: i5, %t_3: i5, %t_4: i5, %t_5: i5, %default: i5, %key: i5) -> (v: i5) {
  %c3_i5 = hw.constant 3 : i5
  %c-16_i5 = hw.constant -16 : i5
  %c-14_i5 = hw.constant -14 : i5
  %c-13_i5 = hw.constant -13 : i5
  %c-9_i5 = hw.constant -9 : i5
  %c-8_i5 = hw.constant -8 : i5
  // CHECK:        [[REG:%.+]] = sv.reg  : !hw.inout<i5>
  // CHECK-NEXT:   [[RES:%.+]] = sv.read_inout [[REG]] : !hw.inout<i5>

  // CHECK-NEXT:   sv.alwayscomb {
  // CHECK-NEXT:     sv.case priority %key : i5
  %10 = comb.icmp eq %key, %c-8_i5 : i5
  %11 = comb.mux %10, %t_5, %9 : i5
  // CHECK-NEXT:     case b11000: {
  // CHECK-NEXT:       sv.bpassign [[REG]], %t_5 : i5
  // CHECK-NEXT:     }

  %8 = comb.icmp eq %key, %c-9_i5 : i5
  %9 = comb.mux %8, %t_4, %7 : i5
  // CHECK-NEXT:     case b10111: {
  // CHECK-NEXT:       sv.bpassign [[REG]], %t_4 : i5
  // CHECK-NEXT:     }

  %6 = comb.icmp eq %key, %c-13_i5 : i5
  %7 = comb.mux %6, %t_3, %5 : i5
  // CHECK-NEXT:     case b10011: {
  // CHECK-NEXT:       sv.bpassign [[REG]], %t_3 : i5
  // CHECK-NEXT:     }

  %4 = comb.icmp eq %key, %c-14_i5 : i5
  %5 = comb.mux %4, %t_2, %3 : i5
  // CHECK-NEXT:     case b10010: {
  // CHECK-NEXT:       sv.bpassign [[REG]], %t_2 : i5
  // CHECK-NEXT:     }

  %2 = comb.icmp eq %key, %c-16_i5 : i5
  %3 = comb.mux %2, %t_1, %1 : i5
  // CHECK-NEXT:     case b10000: {
  // CHECK-NEXT:       sv.bpassign [[REG]], %t_1 : i5
  // CHECK-NEXT:     }

  %0 = comb.icmp eq %key, %c3_i5 : i5
  %1 = comb.mux %0, %t_0, %default : i5
  // CHECK-NEXT:     case b00011: {
  // CHECK-NEXT:       sv.bpassign [[REG]], %t_0 : i5
  // CHECK-NEXT:     }
  // CHECK-NEXT:     default: {
  // CHECK-NEXT:       sv.bpassign [[REG]], %default : i5
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }

  // CHECK-NEXT:   hw.output [[RES]] : i5
  // CHECK-NEXT: }
  hw.output %11 : i5
}