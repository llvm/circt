// RUN: arcilator %s

// Modules with i0 ports and i0 registers must make it through the pipeline:
// the i0 values turn into i0 states, which must be removed like any other i0.

hw.module @UnusedI0Input(in %clock: !seq.clock, in %a: i0, in %x: i8, out y: i8) {
  %reg = seq.compreg %x, %clock : i8
  hw.output %reg : i8
}

hw.module @I0Register(in %clock: !seq.clock, in %a: i0, out b: i0) {
  %reg = seq.compreg %a, %clock : i0
  hw.output %reg : i0
}
