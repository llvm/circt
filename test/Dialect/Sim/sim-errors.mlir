// RUN: circt-opt %s --split-input-file --verify-diagnostics --canonicalize

hw.module @fmt_infinite_concat_verify() {
  %lp = sim.fmt.lit ", {"
  %rp = sim.fmt.lit "}"
  // expected-error @below {{op is infinitely recursive.}}
  %ordinal = sim.fmt.concat (%ordinal, %lp, %ordinal, %rp)
}
// -----

hw.module @fmt_infinite_concat_canonicalize(in %val : i8, out res: !sim.fstring) {
  %c = sim.fmt.char %val : i8
  %0 = sim.fmt.lit "Here we go round the"
  %1 = sim.fmt.lit "prickly pear"
  // expected-warning @below {{Cyclic concatenation detected.}}
  %2 = sim.fmt.concat (%1, %c, %4)
  // expected-warning @below {{Cyclic concatenation detected.}}
  %3 = sim.fmt.concat (%1, %c, %1, %c, %2, %c)
  // expected-warning @below {{Cyclic concatenation detected.}}
  %4 = sim.fmt.concat (%0, %c, %3)
  %5 = sim.fmt.lit "At five o'clock in the morning."
  // expected-warning @below {{Cyclic concatenation detected.}}
  %cat = sim.fmt.concat (%4, %c, %5)
  hw.output %cat : !sim.fstring
}

// -----

hw.module @proc_print_hw() {
  %lit = sim.fmt.lit "Nope"
  // expected-error @below {{must be within a procedural region.}}
  sim.proc.print %lit
}

// -----

sv.macro.decl @SOMEMACRO
hw.module @proc_print_sv() {
  %lit = sim.fmt.lit "Nope"
  sv.ifdef  @SOMEMACRO {
    // expected-error @below {{must be within a procedural region.}}
    sim.proc.print %lit
  }
}

// -----

hw.module.extern @non_func(out arg0: i1, in %arg1: i1, out arg2: i1)

hw.module @dpi_call(in %clock : !seq.clock, in %in: i1) {
  // expected-error @below {{callee must be 'sim.dpi.func' or 'func.func' but got 'hw.module.extern'}}
  %0, %1 = sim.func.call @non_func(%in) : (i1) -> (i1, i1)
}

// -----

sim.func.dpi @mismatch(out arg0: i1, in %arg1: i2, out arg2: i1)

hw.module @dpi_call(in %clock : !seq.clock, in %in: i1) {
  // expected-error @below {{'sim.func.call' op operand type mismatch: expected operand type 'i2', but provided 'i1' for operand number 0}}
  %0, %1 = sim.func.call @mismatch(%in) : (i1) -> (i1, i1)
}


// -----

// expected-note @below {{function result types: 'i1', 'i2'}}
sim.func.dpi @mismatch(out arg0: i1, in %arg1: i1, out arg2: i2)

hw.module @dpi_call(in %clock : !seq.clock, in %in: i1) {
  // expected-error @below {{'sim.func.call' op result type mismatch at index 1}}
  // expected-note @below {{op result types: 'i1', 'i1'}}
  %0, %1 = sim.func.call @mismatch(%in) : (i1) -> (i1, i1)
}
