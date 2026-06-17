// RUN: arcilator --run %s --jit-entry=entry | FileCheck %s --strict-whitespace --match-full-lines
// REQUIRES: arcilator-jit

// The integer formatters pad to the natural width of the value's type using the
// format op's padding character -- a space for `sim.fmt.dec` and '0' for the
// others -- rather than with NUL bytes. A `specifierWidth` of 0 disables padding
// and prints the value in its minimal representation. Values are bracketed so
// the padding is visible to FileCheck under `--strict-whitespace`.
//
// `sim.fmt.bin` is omitted because the arcilator runtime only lowers the
// decimal, hexadecimal, and octal formatters; binary is covered by the
// constant-folder test in test/Dialect/Sim/format-strings.mlir.

func.func @entry() {
  %open = sim.fmt.literal "["
  %close = sim.fmt.literal "]\n"

  %c5_4 = hw.constant 5 : i4
  %c5_8 = hw.constant 5 : i8
  %c106 = hw.constant 106 : i16

  // Default field width: pad to the type's natural width.

  // CHECK:[ 5]
  %dec = sim.fmt.dec %c5_4 : i4
  %decMsg = sim.fmt.concat (%open, %dec, %close)
  sim.proc.print %decMsg

  // CHECK:[05]
  %hex = sim.fmt.hex %c5_8, isUpper false : i8
  %hexMsg = sim.fmt.concat (%open, %hex, %close)
  sim.proc.print %hexMsg

  // CHECK:[000152]
  %oct = sim.fmt.oct %c106 : i16
  %octMsg = sim.fmt.concat (%open, %oct, %close)
  sim.proc.print %octMsg

  // A `specifierWidth` of 0 disables padding for every radix.

  // CHECK:[106]
  %d0 = sim.fmt.dec %c106 specifierWidth 0 : i16
  %d0Msg = sim.fmt.concat (%open, %d0, %close)
  sim.proc.print %d0Msg

  // CHECK:[6a]
  %h0 = sim.fmt.hex %c106, isUpper false specifierWidth 0 : i16
  %h0Msg = sim.fmt.concat (%open, %h0, %close)
  sim.proc.print %h0Msg

  // CHECK:[152]
  %o0 = sim.fmt.oct %c106 specifierWidth 0 : i16
  %o0Msg = sim.fmt.concat (%open, %o0, %close)
  sim.proc.print %o0Msg

  return
}
