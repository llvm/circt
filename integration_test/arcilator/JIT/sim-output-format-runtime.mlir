// RUN: arcilator --run %s --jit-entry=entry | FileCheck --match-full-lines --strict-whitespace %s
// REQUIRES: arcilator-jit

module {
  func.func @entry() {
    %v42 = arith.constant 42 : i8
    %v10 = arith.constant 10 : i8
    %neg = arith.constant -1 : i8
    %zero8 = arith.constant 0 : i8
    %low_unknown = arith.constant 15 : i8
    %high_unknown_z = arith.constant -16 : i8
    %fval = arith.constant 2 : i2
    %funk = arith.constant 1 : i2
    %ch = arith.constant 65 : i8
    %bin = sim.fmt.bin %v10 : i8
    %sep0 = sim.fmt.literal "|"
    %dec = sim.fmt.int 10 0 0 %v42 : i8
    %sep1 = sim.fmt.literal "|"
    %hex = sim.fmt.int 16 4 1 %v42 : i8
    %sep2 = sim.fmt.literal "|"
    %fv = sim.fmt.fvint 2 0 0 %fval, %funk : i2
    %sep3 = sim.fmt.literal "|"
    %signed = sim.fmt.int 10 0 8 %neg : i8
    %endl = sim.fmt.literal "\n"
    %msg = sim.fmt.concat (%bin, %sep0, %dec, %sep1, %hex, %sep2, %fv, %sep3, %signed, %endl)
    // CHECK:{{^00001010[|]42[|]  2A[|]1x[|]-1$}}
    sim.proc.print %msg

    %oct_zero = sim.fmt.int 8 5 4 %v42 : i8
    %sep4 = sim.fmt.literal "|"
    %left_zero = sim.fmt.int 10 5 6 %v42 : i8
    %sep5 = sim.fmt.literal "|"
    %hex_lower_neg = sim.fmt.int 16 0 0 %neg : i8
    %msg2 = sim.fmt.concat (%oct_zero, %sep4, %left_zero, %sep5, %hex_lower_neg, %endl)
    // CHECK:{{^00052[|]42000[|]ff$}}
    sim.proc.print %msg2

    %fv_known_hex = sim.fmt.fvint 16 4 4 %v42, %zero8 : i8
    %sep6 = sim.fmt.literal "|"
    %fv_hex_z = sim.fmt.fvint 16 0 1 %high_unknown_z, %high_unknown_z : i8
    %sep7 = sim.fmt.literal "|"
    %fv_hex_x = sim.fmt.fvint 16 0 0 %zero8, %low_unknown : i8
    %sep8 = sim.fmt.literal "|"
    %fv_dec_z = sim.fmt.fvint 10 0 1 %neg, %neg : i8
    %msg3 = sim.fmt.concat (%fv_known_hex, %sep6, %fv_hex_z, %sep7, %fv_hex_x, %sep8, %fv_dec_z, %endl)
    // CHECK:{{^002a[|]Z0[|]x[|]Z$}}
    sim.proc.print %msg3

    %open = sim.fmt.literal "["
    %close = sim.fmt.literal "]\n"
    %right = sim.fmt.char %ch specifierWidth 3 : i8
    %left = sim.fmt.char %ch isLeftAligned true paddingChar 95 specifierWidth 3 : i8
    %zero = sim.fmt.char %ch paddingChar 48 specifierWidth 3 : i8
    %msg4 = sim.fmt.concat (%open, %right, %close, %open, %left, %close, %open, %zero, %close)
    // CHECK:[  A]
    // CHECK:[A__]
    // CHECK:[00A]
    sim.proc.print %msg4

    sim.proc.timeformat -9, 2, " ns", 0
    %time = arith.constant 1500000 : i64
    %fmt = sim.fmt.time %time, width 0 : i64
    %time_msg = sim.fmt.concat (%fmt, %endl)
    // CHECK:{{^1\.50 ns$}}
    sim.proc.print %time_msg

    %fmt_width = sim.fmt.time %time, width 12 : i64
    %time_width_msg = sim.fmt.concat (%open, %fmt_width, %close)
    // CHECK:[     1.50 ns]
    sim.proc.print %time_width_msg

    return
  }
}
