// RUN: circt-opt --verify-roundtrip --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: hw.module @plusargs_value
hw.module @plusargs_value() {
  // CHECK: sim.plusargs.test "foo"
  %0 = sim.plusargs.test "foo"
  // CHECK: sim.plusargs.value "bar" : i5
  %1, %2 = sim.plusargs.value "bar" : i5
}

// CHECK-LABEL: sim.func.dpi @dpi(out arg0 : i1, in %arg1 : i1, return ret : i1)
sim.func.dpi @dpi(out arg0: i1, in %arg1: i1, return ret: i1)
// CHECK-LABEL: sim.func.dpi @dpi_inout(in %arg0 : i1, inout %arg1 : i1)
sim.func.dpi @dpi_inout(in %arg0: i1, inout %arg1: i1)
func.func private @func(%arg1: i1) -> (i1, i1)

// CHECK-LABEL: hw.module @dpi_call
hw.module @dpi_call(in %clock : !seq.clock, in %enable : i1, in %in: i1) {
  // CHECK: sim.func.dpi.call @dpi(%in) clock %clock enable %enable : (i1) -> (i1, i1)
  %0, %1 = sim.func.dpi.call @dpi(%in) clock %clock enable %enable: (i1) -> (i1, i1)
  // CHECK: sim.func.dpi.call @dpi(%in) clock %clock : (i1) -> (i1, i1)
  %2, %3 = sim.func.dpi.call @dpi(%in) clock %clock : (i1) -> (i1, i1)
  // CHECK: sim.func.dpi.call @func(%in) enable %enable : (i1) -> (i1, i1)
  %4, %5 = sim.func.dpi.call @func(%in) enable %enable : (i1) -> (i1, i1)
  // CHECK: sim.func.dpi.call @func(%in) : (i1) -> (i1, i1)
  %6, %7 = sim.func.dpi.call @func(%in) : (i1) -> (i1, i1)
  // CHECK: sim.func.dpi.call @dpi_inout(%in, %in) : (i1, i1) -> i1
  %8 = sim.func.dpi.call @dpi_inout(%in, %in) : (i1, i1) -> i1
}

// Round-trip tests for !sim.dpi_functy type syntax.
// CHECK-LABEL: func.func @dpi_functy_roundtrip
func.func @dpi_functy_roundtrip(
  // CHECK-SAME: %arg0: !sim.dpi_functy<in "a" : i32, out "b" : i32>
  %arg0: !sim.dpi_functy<in "a" : i32, out "b" : i32>,
  // CHECK-SAME: %arg1: !sim.dpi_functy<in "x" : i8, inout "y" : i16, return "r" : i32>
  %arg1: !sim.dpi_functy<in "x" : i8, inout "y" : i16, return "r" : i32>,
  // CHECK-SAME: %arg2: !sim.dpi_functy<ref "p" : !llvm.ptr>
  %arg2: !sim.dpi_functy<ref "p" : !llvm.ptr>,
  // CHECK-SAME: %arg3: !sim.dpi_functy<>
  %arg3: !sim.dpi_functy<>
) {
  return
}

// CHECK-LABEL: hw.module @GraphSimulationControl
hw.module @GraphSimulationControl(in %clock: !seq.clock, in %en: i1) {
  // CHECK: sim.clocked_terminate %clock, %en, success, verbose
  sim.clocked_terminate %clock, %en, success, verbose
  // CHECK: sim.clocked_terminate %clock, %en, success, quiet
  sim.clocked_terminate %clock, %en, success, quiet
  // CHECK: sim.clocked_terminate %clock, %en, failure, verbose
  sim.clocked_terminate %clock, %en, failure, verbose
  // CHECK: sim.clocked_terminate %clock, %en, failure, quiet
  sim.clocked_terminate %clock, %en, failure, quiet

  // CHECK: sim.clocked_pause %clock, %en, verbose
  sim.clocked_pause %clock, %en, verbose
  // CHECK: sim.clocked_pause %clock, %en, quiet
  sim.clocked_pause %clock, %en, quiet
}

// CHECK-LABEL: func.func @SimulationControl
func.func @SimulationControl() {
  // CHECK: sim.terminate success, verbose
  sim.terminate success, verbose
  // CHECK: sim.terminate success, quiet
  sim.terminate success, quiet
  // CHECK: sim.terminate failure, verbose
  sim.terminate failure, verbose
  // CHECK: sim.terminate failure, quiet
  sim.terminate failure, quiet

  // CHECK: sim.pause verbose
  sim.pause verbose
  // CHECK: sim.pause quiet
  sim.pause quiet
  return
}

// CHECK-LABEL: func.func @FormatStrings
func.func @FormatStrings() {
  // CHECK: sim.fmt.current_time
  sim.fmt.current_time
  // CHECK: sim.fmt.hier_path
  sim.fmt.hier_path
  // CHECK: sim.fmt.hier_path escaped
  sim.fmt.hier_path escaped
  return
}

// CHECK-LABEL: func.func @DynamicStrings
func.func @DynamicStrings(%idx: i32) {
  // CHECK: sim.string.literal "Hello"
  %str = sim.string.literal "Hello"
  // CHECK: sim.string.length
  %len = sim.string.length %str
  // CHECK: sim.string.concat
  %concat = sim.string.concat (%str, %str)
  // CHECK: sim.string.get
  %char = sim.string.get %str[%idx]
  // CHECK: sim.string.string_to_int
  %int = sim.string.string_to_int %str : i32
  return
}

// CHECK-LABEL: hw.module @ProceduralPrintWithGetFile
hw.module @ProceduralPrintWithGetFile(in %clock: !seq.clock, in %condition: i1, in %idx: i32) {
  // CHECK: %[[FMT:.*]] = sim.fmt.literal "literal string"
  %str = sim.fmt.literal "literal string"
  // CHECK: %[[FN0:.*]] = sim.fmt.literal "output_"
  %fn0 = sim.fmt.literal "output_"
  // CHECK: %[[FN1:.*]] = sim.fmt.dec %idx : i32
  %fn1 = sim.fmt.dec %idx : i32
  // CHECK: %[[FN2:.*]] = sim.fmt.literal ".txt"
  %fn2 = sim.fmt.literal ".txt"
  // CHECK: %[[FNAME:.*]] = sim.fmt.concat (%[[FN0]], %[[FN1]], %[[FN2]])
  %fileName = sim.fmt.concat (%fn0, %fn1, %fn2)
  // CHECK: sim.print %[[FMT]] on %clock if %condition
  sim.print %str on %clock if %condition
  // CHECK: sim.triggered %clock if %condition {
  sim.triggered %clock if %condition {
    // CHECK: %[[FILE:.*]] = sim.get_file %[[FNAME]]
    %file = sim.get_file %fileName
    // CHECK: sim.proc.print %[[FMT]] to %[[FILE]]
    sim.proc.print %str to %file
    // CHECK: sim.flush %[[FILE]]
    sim.flush %file
  }
}

// CHECK-LABEL: hw.module @ProceduralPrint
hw.module @ProceduralPrint(in %trigger: i1, in %condition: i1) {
// CHECK-NEXT: hw.triggered
  hw.triggered posedge %trigger (%condition) : i1 {
  ^bb0(%c : i1):
    // CHECK-COUNT-3: sim.fmt.literal
    %foo = sim.fmt.literal "foo"
    %bar = sim.fmt.literal "bar"
    %baz = sim.fmt.literal "baz"
    // CHECK-NEXT: sim.proc.print
    sim.proc.print %foo
    // CHECK-NEXT: scf.if
    scf.if %c {
      // CHECK: sim.proc.print
      sim.proc.print %bar
      // CHECK: else
    } else {
      // CHECK: sim.proc.print
      sim.proc.print %baz
    }
  }
}

// CHECK-LABEL: hw.module @SimTriggered
hw.module @SimTriggered(in %clock: !seq.clock, in %condition: i1) {
  // CHECK: %[[MSG0:.*]] = sim.fmt.literal "tick"
  %msg0 = sim.fmt.literal "tick"
  // CHECK: %[[MSG1:.*]] = sim.fmt.literal "tock"
  %msg1 = sim.fmt.literal "tock"

  // CHECK: sim.triggered %clock {
  sim.triggered %clock {
    // CHECK: sim.proc.print %[[MSG0]]
    sim.proc.print %msg0
  }

  // CHECK: sim.triggered %clock if %condition {
  sim.triggered %clock if %condition {
    // CHECK: sim.proc.print %[[MSG1]]
    sim.proc.print %msg1
  }
}

// CHECK-LABEL: hw.module @StdoutAndStderr
hw.module @StdoutAndStderr(in %clock: !seq.clock, in %condition: i1) {
  // CHECK: %[[STDOUT_STR:.*]] = sim.fmt.literal "Hello, stdout!"
  // CHECK: %[[STDERR_STR:.*]] = sim.fmt.literal "Hello, stderr!"
  %stdout_str = sim.fmt.literal "Hello, stdout!"
  %stderr_str = sim.fmt.literal "Hello, stderr!"
  // CHECK: %[[STDOUT:.*]] = sim.stdout_stream
  // CHECK: %[[STDERR:.*]] = sim.stderr_stream
  %stdout = sim.stdout_stream
  %stderr = sim.stderr_stream
  // CHECK: sim.print %[[STDOUT_STR]] on %clock if %condition to %[[STDOUT]]
  // CHECK: sim.print %[[STDERR_STR]] on %clock if %condition to %[[STDERR]]
  sim.print %stdout_str on %clock if %condition to %stdout
  sim.print %stderr_str on %clock if %condition to %stderr
}

// CHECK-LABEL: hw.module @FormatString(in %str : !sim.dstring)
hw.module @FormatString(in %str: !sim.dstring) {
  // CHECK: sim.fmt.string %str : !sim.dstring
  %fmt0 = sim.fmt.string %str : !sim.dstring
  // CHECK: sim.fmt.string %str isLeftAligned true paddingChar 48 specifierWidth 8 : !sim.dstring
  %fmt1 = sim.fmt.string %str isLeftAligned true paddingChar 48 specifierWidth 8 : !sim.dstring
}
