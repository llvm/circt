// RUN: circt-synth %s -output-longest-path=%t -top counter && cat %t | FileCheck %s
// CHECK-LABEL: # Longest Path Analysis result for "counter"
// CHECK-NEXT: Found 89 closed paths
// CHECK:      Level = 0  . Count = 16 . 50.00   %
// CHECK-NEXT: Level = 1  . Count = 1  . 53.12   %
// CHECK-NEXT: Level = 3  . Count = 1  . 56.25   %
// CHECK-NEXT: Level = 7  . Count = 1  . 59.38   %
// CHECK-NEXT: Level = 11 . Count = 1  . 62.50   %
// CHECK-NEXT: Level = 13 . Count = 1  . 65.62   %
// CHECK-NEXT: Level = 17 . Count = 1  . 68.75   %
// CHECK-NEXT: Level = 19 . Count = 1  . 71.88   %
// CHECK-NEXT: Level = 23 . Count = 1  . 75.00   %
// CHECK-NEXT: Level = 25 . Count = 1  . 78.12   %
// CHECK-NEXT: Level = 27 . Count = 1  . 81.25   %
// CHECK-NEXT: Level = 30 . Count = 1  . 84.38   %
// CHECK-NEXT: Level = 32 . Count = 1  . 87.50   %
// CHECK-NEXT: Level = 35 . Count = 1  . 90.62   %
// CHECK-NEXT: Level = 37 . Count = 1  . 93.75   %
// CHECK-NEXT: Level = 38 . Count = 1  . 96.88   %
// CHECK-NEXT: Level = 48 . Count = 1  . 100.00  %

// CHECK: nyako
hw.module @passthrough(in %a: i16, out result: i16) {
    hw.output %a : i16
}

hw.module @counter(in %a: i16, in %clk: !seq.clock, out result: i16) {
    %reg = seq.compreg %add, %clk : i16
    %add = comb.mul %reg, %a : i16
    %result = hw.instance "passthrough" @passthrough(a: %add: i16) -> (result: i16)
    hw.output %result : i16
}
