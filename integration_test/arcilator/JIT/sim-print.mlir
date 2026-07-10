// RUN: arcilator --run %s > %t.out 2> %t.err
// RUN: FileCheck %s --check-prefix=STDOUT --input-file=%t.out
// RUN: FileCheck %s --check-prefix=STDERR --input-file=%t.err
// RUN: not grep -q stderr-stream %t.out
// RUN: not grep -q stdout-stream %t.err
// REQUIRES: arcilator-jit

module {
  func.func @entry() {
    %endl = sim.fmt.literal "\n"

    // STDOUT: xyz
    %l0 = sim.fmt.literal "xyz"
    sim.proc.print %l0
    sim.proc.print %endl

    // STDOUT: {{ *}}123
    %c123 = hw.constant 123 : i32
    %d0 = sim.fmt.dec %c123 : i32
    sim.proc.print %d0
    sim.proc.print %endl

    // STDOUT: {{ *}}-123
    %cm123 = hw.constant -123 : i32
    %d1 = sim.fmt.dec %cm123 signed : i32
    sim.proc.print %d1
    sim.proc.print %endl

    // STDOUT: {{0*}}7B
    %h0 = sim.fmt.hex %c123, isUpper true : i32
    sim.proc.print %h0
    sim.proc.print %endl

    // STDOUT: {{0*}}7b
    %h1 = sim.fmt.hex %c123, isUpper false : i32
    sim.proc.print %h1
    sim.proc.print %endl

    // STDOUT: {{0*}}173
    %o0 = sim.fmt.oct %c123 : i32
    sim.proc.print %o0
    sim.proc.print %endl

    // STDOUT: {{0*}}1111011
    %b0 = sim.fmt.bin %c123 : i32
    sim.proc.print %b0
    sim.proc.print %endl

    // STDOUT: {
    %c123_8 = hw.constant 123 : i8
    %ch0 = sim.fmt.char %c123_8 : i8
    sim.proc.print %ch0
    sim.proc.print %endl

    // STDOUT: val=
    // STDOUT-LITERAL-SAME: 123,
    // STDOUT-LITERAL-SAME: hex=7b
    %cat0 = sim.fmt.concat (%l0, %d0, %h1)
    %l_val = sim.fmt.literal "val="
    %l_hex = sim.fmt.literal ", hex="
    %cat1 = sim.fmt.concat (%l_val, %d0, %l_hex, %h1)
    sim.proc.print %cat1
    sim.proc.print %endl

    // STDOUT: stdout-stream
    %stdout = sim.stdout_stream
    %stdout_msg = sim.fmt.literal "stdout-stream\n"
    sim.proc.print %stdout_msg to %stdout

    // STDERR: stderr-stream
    %stderr = sim.stderr_stream
    %stderr_msg = sim.fmt.literal "stderr-stream\n"
    sim.proc.print %stderr_msg to %stderr

    return
  }
}
