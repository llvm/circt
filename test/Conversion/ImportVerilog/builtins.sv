// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

function void dummyA(int x); endfunction
function void dummyB(real x); endfunction

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
// CHECK-SAME: [[R:%.+]]: !moore.real
function void DisplayAndSeverityBuiltins(int x, real r);
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

  // CHECK: moore.fmt.int binary [[X]], width 32, align right, pad zero : i32
  $write("%b", x);
  // CHECK: moore.fmt.int binary [[X]], width 32, align right, pad zero : i32
  $write("%B", x);
  // CHECK: moore.fmt.int binary [[X]], width 0, align right, pad zero : i32
  $write("%0b", x);
  // CHECK: moore.fmt.int binary [[X]], width 42, align right, pad zero : i32
  $write("%42b", x);
  // CHECK: moore.fmt.int binary [[X]], width 42, align left, pad zero : i32
  $write("%-42b", x);

  // CHECK: moore.fmt.int octal [[X]], width 11, align right, pad zero : i32
  $write("%o", x);
  // CHECK: moore.fmt.int octal [[X]], width 11, align right, pad zero : i32
  $write("%O", x);
  // CHECK: moore.fmt.int octal [[X]], width 0, align right, pad zero : i32
  $write("%0o", x);
  // CHECK: moore.fmt.int octal [[X]], width 19, align right, pad zero : i32
  $write("%19o", x);
  // CHECK: moore.fmt.int octal [[X]], width 19, align left, pad zero : i32
  $write("%-19o", x);

  // CHECK: moore.fmt.real float [[R]], width 64, align left, pad space : real
  $write("%f", r);
  // CHECK: [[XR:%.+]] = moore.conversion [[X]] : !moore.i32 -> !moore.real
  // CHECK: [[TMP:%.+]] = moore.fmt.real float [[XR]], width 64, align left, pad space : real
  // CHECK: moore.builtin.display [[TMP]]
  $write("%f", x);

  // CHECK: moore.fmt.int decimal [[X]], width 10, align right, pad space : i32
  $write("%d", x);
  // CHECK: moore.fmt.int decimal [[X]], width 10, align right, pad space : i32
  $write("%D", x);
  // CHECK: moore.fmt.int decimal [[X]], width 0, align right, pad space : i32
  $write("%0d", x);
  // CHECK: moore.fmt.int decimal [[X]], width 19, align right, pad space : i32
  $write("%19d", x);
  // CHECK: moore.fmt.int decimal [[X]], width 19, align left, pad space : i32
  $write("%-19d", x);

  // CHECK: moore.fmt.int hex_lower [[X]], width 8, align right, pad zero : i32
  $write("%h", x);
  // CHECK: moore.fmt.int hex_lower [[X]], width 8, align right, pad zero : i32
  $write("%x", x);
  // CHECK: moore.fmt.int hex_upper [[X]], width 8, align right, pad zero : i32
  $write("%H", x);
  // CHECK: moore.fmt.int hex_upper [[X]], width 8, align right, pad zero : i32
  $write("%X", x);
  // CHECK: moore.fmt.int hex_lower [[X]], width 0, align right, pad zero : i32
  $write("%0h", x);
  // CHECK: moore.fmt.int hex_lower [[X]], width 19, align right, pad zero : i32
  $write("%19h", x);
  // CHECK: moore.fmt.int hex_lower [[X]], width 19, align right, pad zero : i32
  $write("%019h", x);
  // CHECK: moore.fmt.int hex_lower [[X]], width 19, align left, pad zero : i32
  $write("%-19h", x);
  // CHECK: moore.fmt.int hex_lower [[X]], width 19, align left, pad zero : i32
  $write("%-019h", x);

  // CHECK: [[TMP:%.+]] = moore.fmt.int decimal [[X]], width 10, align right, pad space : i32
  // CHECK: moore.builtin.display [[TMP]]
  $write(x);
  // CHECK: [[TMP:%.+]] = moore.fmt.int binary [[X]], width 32, align right, pad zero : i32
  // CHECK: moore.builtin.display [[TMP]]
  $writeb(x);
  // CHECK: [[TMP:%.+]] = moore.fmt.int octal [[X]], width 11, align right, pad zero : i32
  // CHECK: moore.builtin.display [[TMP]]
  $writeo(x);
  // CHECK: [[TMP:%.+]] = moore.fmt.int hex_lower [[X]], width 8, align right, pad zero : i32
  // CHECK: moore.builtin.display [[TMP]]
  $writeh(x);

  // CHECK: [[TMP1:%.+]] = moore.fmt.int decimal [[X]], width 10, align right, pad space : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $display(x);
  // CHECK: [[TMP1:%.+]] = moore.fmt.int binary [[X]], width 32, align right, pad zero : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $displayb(x);
  // CHECK: [[TMP1:%.+]] = moore.fmt.int octal [[X]], width 11, align right, pad zero : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $displayo(x);
  // CHECK: [[TMP1:%.+]] = moore.fmt.int hex_lower [[X]], width 8, align right, pad zero : i32
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
  // CHECK: [[TMP:%.+]] = moore.fmt.real
  // CHECK: moore.builtin.severity info [[TMP]]
  $info("%f", r);
  // CHECK: [[TMP:%.+]] = moore.fmt.literal ""
  // CHECK: moore.builtin.severity warning [[TMP]]
  $warning;
  // CHECK: [[TMP:%.+]] = moore.fmt.int
  // CHECK: moore.builtin.severity warning [[TMP]]
  $warning("%d", x);
  // CHECK: [[TMP:%.+]] = moore.fmt.real
  // CHECK: moore.builtin.severity warning [[TMP]]
  $warning("%f", r);
  // CHECK: [[TMP:%.+]] = moore.fmt.literal ""
  // CHECK: moore.builtin.severity error [[TMP]]
  $error;
  // CHECK: [[TMP:%.+]] = moore.fmt.int
  // CHECK: moore.builtin.severity error [[TMP]]
  $error("%d", x);
  // CHECK: [[TMP:%.+]] = moore.fmt.real
  // CHECK: moore.builtin.severity error [[TMP]]
  $error("%f", r);
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
  // CHECK: [[TMP:%.+]] = moore.fmt.real
  // CHECK: moore.builtin.severity fatal [[TMP]]
  // CHECK: moore.unreachable
  if (0) $fatal(1, "%f", r);
endfunction

// IEEE 1800-2017 § 20.8 "Math functions"
// CHECK-LABEL: func.func private @MathBuiltins(
// CHECK-SAME: [[X:%.+]]: !moore.i32
// CHECK-SAME: [[Y:%.+]]: !moore.l42
// CHECK-SAME: [[R:%.+]]: !moore.real
function void MathBuiltins(int x, logic [41:0] y, real r);
  // CHECK: moore.builtin.clog2 [[X]] : i32
  dummyA($clog2(x));
  // CHECK: moore.builtin.clog2 [[Y]] : l42
  dummyA($clog2(y));

  // CHECK:  moore.builtin.ln [[R]] : real
  dummyB($ln(r));
  // CHECK:  moore.builtin.log10 [[R]] : real
  dummyB($log10(r));
  // CHECK:  moore.builtin.exp [[R]] : real
  dummyB($exp(r));
  // CHECK:  moore.builtin.sqrt [[R]] : real
  dummyB($sqrt(r));
  // CHECK:  moore.builtin.floor [[R]] : real
  dummyB($floor(r));
  // CHECK:  moore.builtin.ceil [[R]] : real
  dummyB($ceil(r));
  // CHECK:  moore.builtin.sin [[R]] : real
  dummyB($sin(r));
  // CHECK:  moore.builtin.cos [[R]] : real
  dummyB($cos(r));
  // CHECK:  moore.builtin.tan [[R]] : real
  dummyB($tan(r));
  // CHECK:  moore.builtin.asin [[R]] : real
  dummyB($asin(r));
  // CHECK:  moore.builtin.acos [[R]] : real
  dummyB($acos(r));
  // CHECK:  moore.builtin.atan [[R]] : real
  dummyB($atan(r));
  // CHECK:  moore.builtin.sinh [[R]] : real
  dummyB($sinh(r));
  // CHECK:  moore.builtin.cosh [[R]] : real
  dummyB($cosh(r));
  // CHECK:  moore.builtin.tanh [[R]] : real
  dummyB($tanh(r));
  // CHECK:  moore.builtin.asinh [[R]] : real
  dummyB($asinh(r));
  // CHECK:  moore.builtin.acosh [[R]] : real
  dummyB($acosh(r));
  // CHECK:  moore.builtin.atanh [[R]] : real
  dummyB($atanh(r));
endfunction
