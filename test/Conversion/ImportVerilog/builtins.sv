// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

function void dummyA(int x); endfunction

// IEEE 1800-2017 § 20.2 "Simulation control system tasks"
// CHECK-LABEL: func.func private @SimulationControlBuiltins(
function void SimulationControlBuiltins(bit x);
  // CHECK: moore.builtin.finish_message false
  // CHECK: moore.builtin.stop
  $stop;
  // CHECK-NOT: moore.builtin.finish_message
  // CHECK: moore.builtin.stop
  $stop(0);
  // CHECK: moore.builtin.finish_message true
  // CHECK: moore.builtin.stop
  $stop(2);

  // CHECK: moore.builtin.finish_message false
  // CHECK: moore.builtin.finish 0
  // CHECK: moore.unreachable
  if (x) $finish;
  // CHECK-NOT: moore.builtin.finish_message
  // CHECK: moore.builtin.finish 0
  // CHECK: moore.unreachable
  if (x) $finish(0);
  // CHECK: moore.builtin.finish_message true
  // CHECK: moore.builtin.finish 0
  // CHECK: moore.unreachable
  if (x) $finish(2);

  // Ignore `$exit` until we have support for programs.
  // CHECK-NOT: moore.builtin.finish
  $exit;
endfunction

// IEEE 1800-2017 § 20.10 "Severity tasks"
// IEEE 1800-2017 § 21.2 "Display system tasks"
// CHECK-LABEL: func.func private @DisplayAndSeverityBuiltins(
// CHECK-SAME: [[X:%.+]]: !moore.i32
function void DisplayAndSeverityBuiltins(int x);
  // CHECK: [[ARG0:%.+]] = moore.variable %arg0 : <i32>
  // CHECK: [[TMP:%.+]] = moore.fmt.literal "\0A"
  // CHECK: moore.builtin.display [[TMP]]
  $display;
  // CHECK: [[TMP1:%.+]] = moore.fmt.literal "hello"
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $display("hello");

  // CHECK-NOT: moore.builtin.display
  $write;
  $write(,);
  // CHECK: [[TMP:%.+]] = moore.fmt.literal "hello\0A world \\ foo ! bar % \22"
  // CHECK: moore.builtin.display [[TMP]]
  $write("hello\n world \\ foo \x21 bar %% \042");

  // CHECK: [[TMP1:%.+]] = moore.fmt.literal "foo "
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "bar"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $write("foo %s", "bar");

  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int binary [[READ0]], width 32, align right, pad zero : i32
  $write("%b", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int binary [[READ0]], width 32, align right, pad zero : i32
  $write("%B", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int binary [[READ0]], width 0, align right, pad zero : i32
  $write("%0b", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int binary [[READ0]], width 42, align right, pad zero : i32
  $write("%42b", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int binary [[READ0]], width 42, align left, pad zero : i32
  $write("%-42b", x);

  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int octal [[READ0]], width 11, align right, pad zero : i32
  $write("%o", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int octal [[READ0]], width 11, align right, pad zero : i32
  $write("%O", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int octal [[READ0]], width 0, align right, pad zero : i32
  $write("%0o", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int octal [[READ0]], width 19, align right, pad zero : i32
  $write("%19o", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int octal [[READ0]], width 19, align left, pad zero : i32
  $write("%-19o", x);

  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int decimal [[READ0]], width 10, align right, pad space : i32
  $write("%d", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int decimal [[READ0]], width 10, align right, pad space : i32
  $write("%D", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int decimal [[READ0]], width 0, align right, pad space : i32
  $write("%0d", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int decimal [[READ0]], width 19, align right, pad space : i32
  $write("%19d", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int decimal [[READ0]], width 19, align left, pad space : i32
  $write("%-19d", x);

  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int hex_lower [[READ0]], width 8, align right, pad zero : i32
  $write("%h", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int hex_lower [[READ0]], width 8, align right, pad zero : i32
  $write("%x", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int hex_upper [[READ0]], width 8, align right, pad zero : i32
  $write("%H", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int hex_upper [[READ0]], width 8, align right, pad zero : i32
  $write("%X", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int hex_lower [[READ0]], width 0, align right, pad zero : i32
  $write("%0h", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int hex_lower [[READ0]], width 19, align right, pad zero : i32
  $write("%19h", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int hex_lower [[READ0]], width 19, align right, pad zero : i32
  $write("%019h", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int hex_lower [[READ0]], width 19, align left, pad zero : i32
  $write("%-19h", x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.fmt.int hex_lower [[READ0]], width 19, align left, pad zero : i32
  $write("%-019h", x);

  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: [[TMP:%.+]] = moore.fmt.int decimal [[READ0]], width 10, align right, pad space : i32
  // CHECK: moore.builtin.display [[TMP]]
  $write(x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: [[TMP:%.+]] = moore.fmt.int binary [[READ0]], width 32, align right, pad zero : i32
  // CHECK: moore.builtin.display [[TMP]]
  $writeb(x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: [[TMP:%.+]] = moore.fmt.int octal [[READ0]], width 11, align right, pad zero : i32
  // CHECK: moore.builtin.display [[TMP]]
  $writeo(x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: [[TMP:%.+]] = moore.fmt.int hex_lower [[READ0]], width 8, align right, pad zero : i32
  // CHECK: moore.builtin.display [[TMP]]
  $writeh(x);

  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: [[TMP1:%.+]] = moore.fmt.int decimal [[READ0]], width 10, align right, pad space : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $display(x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: [[TMP1:%.+]] = moore.fmt.int binary [[READ0]], width 32, align right, pad zero : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $displayb(x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: [[TMP1:%.+]] = moore.fmt.int octal [[READ0]], width 11, align right, pad zero : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $displayo(x);
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: [[TMP1:%.+]] = moore.fmt.int hex_lower [[READ0]], width 8, align right, pad zero : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $displayh(x);

  // CHECK: [[TMP:%.+]] = moore.fmt.literal ""
  // CHECK: moore.builtin.severity info [[TMP]]
  $info;
  // CHECK: [[TMP:%.+]] = moore.fmt.int
  // CHECK: moore.builtin.severity info [[TMP]]
  $info("%d", x);
  // CHECK: [[TMP:%.+]] = moore.fmt.literal ""
  // CHECK: moore.builtin.severity warning [[TMP]]
  $warning;
  // CHECK: [[TMP:%.+]] = moore.fmt.int
  // CHECK: moore.builtin.severity warning [[TMP]]
  $warning("%d", x);
  // CHECK: [[TMP:%.+]] = moore.fmt.literal ""
  // CHECK: moore.builtin.severity error [[TMP]]
  $error;
  // CHECK: [[TMP:%.+]] = moore.fmt.int
  // CHECK: moore.builtin.severity error [[TMP]]
  $error("%d", x);
  // CHECK: [[TMP:%.+]] = moore.fmt.literal ""
  // CHECK: moore.builtin.severity fatal [[TMP]]
  // CHECK: moore.builtin.finish_message false
  // CHECK: moore.builtin.finish 1
  // CHECK: moore.unreachable
  if (0) $fatal;
  // CHECK-NOT: moore.builtin.finish_message
  // CHECK: moore.unreachable
  if (0) $fatal(0);
  // CHECK: moore.builtin.finish_message true
  // CHECK: moore.unreachable
  if (0) $fatal(2);
  // CHECK: [[TMP:%.+]] = moore.fmt.int
  // CHECK: moore.builtin.severity fatal [[TMP]]
  // CHECK: moore.unreachable
  if (0) $fatal(1, "%d", x);
endfunction

// IEEE 1800-2017 § 20.8 "Math functions"
// CHECK-LABEL: func.func private @MathBuiltins(
// CHECK-SAME: [[X:%.+]]: !moore.i32
// CHECK-SAME: [[Y:%.+]]: !moore.l42
function void MathBuiltins(int x, logic [41:0] y);
  // CHECK: [[ARG0:%.+]] = moore.variable %arg0 : <i32>
  // CHECK: [[ARG1:%.+]] = moore.variable %arg1 : <l42>
  // CHECK: [[READ0:%.+]] = moore.read [[ARG0]] : <i32>
  // CHECK: moore.builtin.clog2 [[READ0]] : i32
  dummyA($clog2(x));
  // CHECK: [[READ1:%.+]] = moore.read [[ARG1]] : <l42>
  // CHECK: moore.builtin.clog2 [[READ1]] : l42
  dummyA($clog2(y));
endfunction
