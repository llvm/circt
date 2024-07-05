// RUN: circt-opt %s --split-input-file --verify-diagnostics

hw.module @fmt_infinite_concat() {
  %lp = sim.fmt.lit ", {"
  %rp = sim.fmt.lit "}"
  // expected-error @below {{op is infinitely recursive.}}
  %ordinal = sim.fmt.concat (%ordinal, %lp, %ordinal, %rp)
}

// -----

hw.module @proc_print() {
  %lit = sim.fmt.lit "Nope"
  // expected-error @below {{must be within a procedural region.}}
  sim.proc.print (%lit)
}
