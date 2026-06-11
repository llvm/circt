// RUN: arcilator --run %s --jit-entry=entry | FileCheck %s --strict-whitespace --match-full-lines
// REQUIRES: arcilator-jit

// The integer formatters must pad with the format op's padding character -- a
// space for `sim.fmt.dec` and '0' for `sim.fmt.hex` -- rather than with NUL
// bytes. The value is bracketed so the padding is visible to FileCheck under
// `--strict-whitespace`.

func.func @entry() {
  %open = sim.fmt.literal "["
  %close = sim.fmt.literal "]\n"

  // CHECK:[   5]
  %c4 = hw.constant 5 : i4
  %dec = sim.fmt.dec %c4 : i4
  %decMsg = sim.fmt.concat (%open, %dec, %close)
  sim.proc.print %decMsg

  // CHECK:[05]
  %c8 = hw.constant 5 : i8
  %hex = sim.fmt.hex %c8, isUpper false : i8
  %hexMsg = sim.fmt.concat (%open, %hex, %close)
  sim.proc.print %hexMsg

  return
}
