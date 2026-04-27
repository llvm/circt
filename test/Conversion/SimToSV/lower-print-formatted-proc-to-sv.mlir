// RUN: circt-opt --lower-sim-to-sv %s | FileCheck %s

// CHECK: sv.func private @"__circt_lib_logging::FileDescriptor::get"
// CHECK: sv.macro.decl @__CIRCT_LIB_LOGGING
// CHECK: emit.fragment @CIRCT_LIB_LOGGING_FRAGMENT

// This pass assumes sim.proc.print is already in some procedural region.
// It lowers both SV and non-SV procedural containers (e.g. hw.triggered).

// CHECK-LABEL: hw.module @proc_print
hw.module @proc_print(in %clk : i1, in %arg: i8) {
  // CHECK: hw.triggered posedge %clk
  hw.triggered posedge %clk (%arg) : i8 {
    ^bb0(%arg_in : i8):
    %l0 = sim.fmt.literal "err: "
    %f0 = sim.fmt.hex %arg_in, isUpper true specifierWidth 2 : i8
    %l1 = sim.fmt.literal " 100%"
    %msg = sim.fmt.concat (%l0, %f0, %l1)

    // CHECK: sv.write "err: %02X 100%%"
    sim.proc.print %msg
  }
}


// CHECK-LABEL: hw.module @all_format_fragments
hw.module @all_format_fragments(
    in %clk : i1, in %ival : i16, in %ch : i8, in %fval : f64) {
  hw.triggered posedge %clk (%ival, %ch, %fval) : i16, i8, f64 {
    ^bb0(%ival_in : i16, %ch_in : i8, %fval_in : f64):
    %i0 = sim.fmt.literal "dec="
    %f0 = sim.fmt.dec %ival_in specifierWidth 6 signed : i16
    %i1 = sim.fmt.literal " hex="
    %f1 = sim.fmt.hex %ival_in, isUpper true paddingChar 48 specifierWidth 4 : i16
    %i2 = sim.fmt.literal " oct="
    %f2 = sim.fmt.oct %ival_in isLeftAligned true specifierWidth 6 : i16
    %i3 = sim.fmt.literal " bin="
    %f3 = sim.fmt.bin %ival_in paddingChar 32 specifierWidth 8 : i16
    %i4 = sim.fmt.literal " char="
    %f4 = sim.fmt.char %ch_in : i8
    %i5 = sim.fmt.literal " exp="
    %f5 = sim.fmt.exp %fval_in fieldWidth 10 fracDigits 3 : f64
    %i6 = sim.fmt.literal " flt="
    %f6 = sim.fmt.flt %fval_in isLeftAligned true fieldWidth 8 fracDigits 2 : f64
    %i7 = sim.fmt.literal " gen="
    %f7 = sim.fmt.gen %fval_in fracDigits 4 : f64
    %i8 = sim.fmt.literal " path="
    %f8 = sim.fmt.hier_path
    %i9 = sim.fmt.literal " esc="
    %f9 = sim.fmt.hier_path escaped
    %i10 = sim.fmt.literal " pct=%"
    %msg = sim.fmt.concat (%i0, %f0, %i1, %f1, %i2, %f2, %i3, %f3, %i4, %f4, %i5, %f5, %i6, %f6, %i7, %f7, %i8, %f8, %i9, %f9, %i10)

    // CHECK: ^bb0(%[[IVAL:.+]]: i16, %[[CH:.+]]: i8, %[[FVAL:.+]]: f64):
    // CHECK-NEXT: %[[SIGNED:.+]] = sv.system "signed"(%[[IVAL]]) : (i16) -> i16
    // CHECK-NEXT: sv.write "dec=%6d hex=%04X oct=%-06o bin=%8b char=%c exp=%10.3e flt=%-8.2f gen=%.4g path=%m esc=%M pct=%%"(%[[SIGNED]], %[[IVAL]], %[[IVAL]], %[[IVAL]], %[[CH]], %[[FVAL]], %[[FVAL]], %[[FVAL]]) : i16, i16, i16, i16, i8, f64, f64, f64
    sim.proc.print %msg
  }
}

// CHECK-LABEL: hw.module @nested_concat_order
hw.module @nested_concat_order(in %clk : i1, in %lhs : i8, in %rhs : i8) {
  hw.triggered posedge %clk (%lhs, %rhs) : i8, i8 {
    ^bb0(%lhs_in : i8, %rhs_in : i8):
    %l0 = sim.fmt.literal "L="
    %l1 = sim.fmt.literal ", R="
    %d0 = sim.fmt.dec %lhs_in specifierWidth 3 : i8
    %h0 = sim.fmt.hex %rhs_in, isUpper false specifierWidth 2 : i8

    %c0 = sim.fmt.concat (%l0, %d0)
    %c1 = sim.fmt.concat (%c0, %l1)
    %c2 = sim.fmt.concat (%c1, %h0)

    // CHECK: ^bb0(%[[LHS:.+]]: i8, %[[RHS:.+]]: i8):
    // CHECK-NEXT: %[[UNSIGNED:.+]] = sv.system "unsigned"(%[[LHS]]) : (i8) -> i8
    // CHECK-NEXT: sv.write "L=%3d, R=%02x"(%[[UNSIGNED]], %[[RHS]]) : i8, i8
    sim.proc.print %c2
  }
}

// CHECK-LABEL: hw.module @dce_uses_outer_procedural_root
hw.module @dce_uses_outer_procedural_root(
    in %clk : i1, in %cond : i1, in %val : i8) {
  // CHECK: hw.triggered posedge %clk
  hw.triggered posedge %clk (%cond, %val) : i1, i8 {
    ^bb0(%cond_in : i1, %val_in : i8):
    %lit = sim.fmt.literal "v="
    %fmt = sim.fmt.dec %val_in : i8
    %msg = sim.fmt.concat (%lit, %fmt)
    // CHECK: ^bb0(%[[COND:.+]]: i1, %[[VAL:.+]]: i8):
    // CHECK: scf.if %[[COND]] {
    // CHECK: %[[UNSIGNED:.+]] = sv.system "unsigned"(%[[VAL]]) : (i8) -> i8
    // CHECK-NEXT: sv.write "v=%d"(%[[UNSIGNED]]) : i8
    scf.if %cond_in {
      sim.proc.print %msg
    }
  }
}

hw.module @triggered_print(in %clk : i1) {
  // CHECK-LABEL: hw.module @triggered_print
  hw.triggered posedge %clk {
    %lit = sim.fmt.literal "hello"
    // CHECK: sv.write "hello"
    sim.proc.print %lit
  }
}

hw.module @print_to_stdout_and_stderr(in %clk : i1) {
  // CHECK-LABEL: hw.module @print_to_stdout_and_stderr
  hw.triggered posedge %clk {
    %stdout = sim.stdout_stream
    %stderr = sim.stderr_stream
    %lit = sim.fmt.literal "literal"
    // CHECK-DAG: %[[STDOUT:.+]] = hw.constant -2147483647 : i32
    // CHECK-DAG: %[[STDERR:.+]] = hw.constant -2147483646 : i32
    // CHECK: sv.fwrite %[[STDOUT]], "literal"
    sim.proc.print %lit to %stdout
    // CHECK-NEXT: sv.fwrite %[[STDERR]], "literal"
    sim.proc.print %lit to %stderr
  }
}

hw.module @print_to_file(in %clk : i1, in %idx : i8) {
  // CHECK-LABEL: hw.module @print_to_file
  // CHECK-SAME: emit.fragments = [@CIRCT_LIB_LOGGING_FRAGMENT]
  hw.triggered posedge %clk (%idx) : i8 {
    ^bb0(%idx_in : i8):
    %filePrefix = sim.fmt.literal "trace_"
    %fileIndex = sim.fmt.dec %idx_in paddingChar 48 specifierWidth 2 : i8
    %fileSuffix = sim.fmt.literal ".log"
    %fileName = sim.fmt.concat (%filePrefix, %fileIndex, %fileSuffix)
    %file = sim.get_file %fileName
    %msgPrefix = sim.fmt.literal "value="
    %msgValue = sim.fmt.hex %idx_in, isUpper false specifierWidth 2 : i8
    %msg = sim.fmt.concat (%msgPrefix, %msgValue)
    // CHECK: ^bb0(%[[IDX:.+]]: i8):
    // CHECK-NEXT: %[[UNSIGNED:.+]] = sv.system "unsigned"(%[[IDX]]) : (i8) -> i8
    // CHECK-NEXT: %[[FILENAME:.+]] = sv.sformatf "trace_%02d.log"(%[[UNSIGNED]]) : i8
    // CHECK-NEXT: %[[FD:.+]] = sv.func.call.procedural @"__circt_lib_logging::FileDescriptor::get"(%[[FILENAME]]) : (!hw.string) -> i32
    // CHECK-NEXT: sv.fwrite %[[FD]], "value=%02x"(%[[IDX]]) : i8
    sim.proc.print %msg to %file
  }
}

hw.module @print_to_file_under_condition(in %clk : i1, in %idx : i8, in %en : i1) {
  // CHECK-LABEL: hw.module @print_to_file_under_condition
  // CHECK-SAME: emit.fragments = [@CIRCT_LIB_LOGGING_FRAGMENT]
  hw.triggered posedge %clk (%idx, %en) : i8, i1 {
    ^bb0(%idx_in : i8, %en_in : i1):
    %filePrefix = sim.fmt.literal "trace_"
    %fileIndex = sim.fmt.dec %idx_in paddingChar 48 specifierWidth 2 : i8
    %fileSuffix = sim.fmt.literal ".log"
    %fileName = sim.fmt.concat (%filePrefix, %fileIndex, %fileSuffix)
    %file = sim.get_file %fileName
    scf.if %en_in {
      %msg = sim.fmt.literal "enabled"
      // CHECK: ^bb0(%[[IDX:.+]]: i8, %[[EN:.+]]: i1):
      // CHECK-NEXT: scf.if %[[EN]] {
      // CHECK-NEXT:   %[[UNSIGNED:.+]] = sv.system "unsigned"(%[[IDX]]) : (i8) -> i8
      // CHECK-NEXT:   %[[FILENAME:.+]] = sv.sformatf "trace_%02d.log"(%[[UNSIGNED]]) : i8
      // CHECK-NEXT:   %[[FD:.+]] = sv.func.call.procedural @"__circt_lib_logging::FileDescriptor::get"(%[[FILENAME]]) : (!hw.string) -> i32
      // CHECK-NEXT:   sv.fwrite %[[FD]], "enabled"
      sim.proc.print %msg to %file
    }
  }
}
