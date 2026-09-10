// RUN: arcilator %s

// This test ensures that the pipeline completes, without checking the result.
// This is to make sure compatibility for some constructs does not regress.

hw.module @arbitrary_reset_value(in %value: i32, in %clock: !seq.clock, in %reset: i1, in %resetValue: i32, out out: i32) {
    %reg = seq.compreg %value, %clock reset %reset, %resetValue : i32
    hw.output %reg : i32
}
