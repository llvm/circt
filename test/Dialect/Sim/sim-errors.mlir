// RUN: circt-opt %s --split-input-file --verify-diagnostics --canonicalize

hw.module @fmt_infinite_concat_verify() {
  %lp = sim.fmt.literal ", {"
  %rp = sim.fmt.literal "}"
  // expected-error @below {{op is infinitely recursive.}}
  %ordinal = sim.fmt.concat (%ordinal, %lp, %ordinal, %rp)
}
// -----

hw.module @fmt_infinite_concat_canonicalize(in %val : i8, out res: !sim.fstring) {
  %c = sim.fmt.char %val : i8
  %0 = sim.fmt.literal "Here we go round the"
  %1 = sim.fmt.literal "prickly pear"
  // expected-warning @below {{Cyclic concatenation detected.}}
  %2 = sim.fmt.concat (%1, %c, %4)
  // expected-warning @below {{Cyclic concatenation detected.}}
  %3 = sim.fmt.concat (%1, %c, %1, %c, %2, %c)
  // expected-warning @below {{Cyclic concatenation detected.}}
  %4 = sim.fmt.concat (%0, %c, %3)
  %5 = sim.fmt.literal "At five o'clock in the morning."
  // expected-warning @below {{Cyclic concatenation detected.}}
  %cat = sim.fmt.concat (%4, %c, %5)
  hw.output %cat : !sim.fstring
}

// -----

hw.module @proc_print_hw() {
  %lit = sim.fmt.literal "Nope"
  // expected-error @below {{must be within a procedural region.}}
  sim.proc.print %lit
}

// -----

sv.macro.decl @SOMEMACRO
hw.module @proc_print_sv() {
  %lit = sim.fmt.literal "Nope"
  sv.ifdef  @SOMEMACRO {
    // expected-error @below {{must be within a procedural region.}}
    sim.proc.print %lit
  }
}

// -----

hw.module.extern @non_func(out arg0: i1, in %arg1: i1, out arg2: i1)

hw.module @dpi_call(in %clock : !seq.clock, in %in: i1) {
  // expected-error @below {{callee must be 'sim.func.dpi' or 'func.func' but got 'hw.module.extern'}}
  %0, %1 = sim.func.dpi.call @non_func(%in) : (i1) -> (i1, i1)
}

// -----

// expected-error @below {{'return' argument must be the last argument}}
sim.func.dpi @dpi_bad_return(return ret: i1, out other: i1)

// -----

sim.func.dpi @dpi_sig(in %a: i1, return ret: i1)
hw.module @dpi_bad_call_arity(in %clock : !seq.clock, in %in: i1) {
  // expected-error @below {{expects 1 DPI results, but got 2}}
  %0, %1 = sim.func.dpi.call @dpi_sig(%in) : (i1) -> (i1, i1)
}

// -----

sim.func.dpi @dpi_inout(in %a: i1, inout %state: i8)
hw.module @dpi_bad_call_types(in %clock : !seq.clock, in %in: i1) {
  // expected-error @below {{operand type mismatch: expected 'i8', but got 'i1'}}
  %0 = sim.func.dpi.call @dpi_inout(%in, %in) : (i1, i1) -> i8
}

// -----

// expected-error @below {{'ref' arguments must use !llvm.ptr type}}
sim.func.dpi @dpi_bad_ref_type(ref %arg : i32)

// -----

hw.module @queue_concat(in %q1: !sim.queue<i32, 0>, in %q2: !sim.queue<i16, 0>) {
  // expected-error @below {{'sim.queue.concat' op sim::Queue element type 'i16' doesn't match result sim::Queue element type 'i32'}}
  sim.queue.concat (%q1, %q2) : (!sim.queue<i32, 0>, !sim.queue<i16, 0>) <i32, 5>
}

hw.module @queue_from_array(in %uparr: !hw.array<5xi33>) {
  // expected-error @below {{'sim.queue.from_array' op sim::Queue element type 'i32' doesn't match hw::ArrayType element type 'i33'}}
  sim.queue.from_array %uparr : !hw.array<5xi33> -> <i32, 0>
}
