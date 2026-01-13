// RUN: arcilator --run %s | FileCheck %s
// REQUIRES: arcilator-jit

module {
  func.func @entry() {
    // CHECK: xyz
    %l0 = sim.fmt.literal "xyz"
    sim.proc.print %l0

    // CHECK: {{ *}}123
    %c123 = hw.constant 123 : i32
    %d0 = sim.fmt.dec %c123 : i32
    sim.proc.print %d0

    // CHECK: {{ *}}-123
    %cm123 = hw.constant -123 : i32
    %d1 = sim.fmt.dec %cm123 signed : i32
    sim.proc.print %d1

    // CHECK: {{0*}}7B
    %h0 = sim.fmt.hex %c123, isUpper true : i32
    sim.proc.print %h0

    // CHECK: {{0*}}7b
    %h1 = sim.fmt.hex %c123, isUpper false : i32
    sim.proc.print %h1

    // CHECK: {{0*}}173
    %o0 = sim.fmt.oct %c123 : i32
    sim.proc.print %o0

    // CHECK: {
    %c123_8 = hw.constant 123 : i8
    %ch0 = sim.fmt.char %c123_8 : i8
    sim.proc.print %ch0

    // CHECK: val={{ *}}123, hex={{0*}}7b
    %cat0 = sim.fmt.concat (%l0, %d0, %h1)
    %l_val = sim.fmt.literal "val="
    %l_hex = sim.fmt.literal ", hex="
    %cat1 = sim.fmt.concat (%l_val, %d0, %l_hex, %h1)
    sim.proc.print %cat1


    return
  }
}
