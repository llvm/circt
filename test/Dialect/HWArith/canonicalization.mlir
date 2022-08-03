// RUN: circt-opt %s -canonicalize | FileCheck %s

// CHECK: hw.module @test1(%a: ui8) -> (out: ui8) {
// CHECK:   hw.output %a : ui8
hw.module @test1(%a : ui8) -> (out: ui8) {
    %0 = hwarith.cast %a : (ui8) -> (ui8)
    hw.output %0 : ui8
}

// CHECK: hw.module @test2(%a: ui8) -> (out: si8) {
// CHECK:   %0 = hwarith.cast %a : (ui8) -> si8
// CHECK:   hw.output %0 : si8
hw.module @test2(%a : ui8) -> (out: si8) {
    %0 = hwarith.cast %a : (ui8) -> (i8)
    %1 = hwarith.cast %0 : (i8) -> (si8)
    hw.output %1 : si8
}
