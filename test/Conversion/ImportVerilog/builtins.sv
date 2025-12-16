// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

function void dummyA(int x); endfunction
function void dummyB(real x); endfunction
function void dummyC(shortreal x); endfunction
function void dummyD(string x); endfunction
function void dummyE(byte x); endfunction

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
// CHECK-SAME: [[R:%.+]]: !moore.f64
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

  // CHECK: moore.fmt.real float [[R]]
  $write("%f", r);
  // CHECK: [[XR:%.+]] = moore.sint_to_real [[X]] : i32 -> f64
  // CHECK: [[TMP:%.+]] = moore.fmt.real float [[XR]]
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

  // CHECK: [[TMP1:%.+]] = moore.fmt.real float [[R]]
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $display("%f", r);

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
// CHECK-SAME: [[R:%.+]]: !moore.f64
function void MathBuiltins(int x, logic [41:0] y, real r);
  // CHECK: moore.builtin.clog2 [[X]] : i32
  dummyA($clog2(x));
  // CHECK: moore.builtin.clog2 [[Y]] : l42
  dummyA($clog2(y));

  // CHECK:  moore.builtin.ln [[R]] : f64
  dummyB($ln(r));
  // CHECK:  moore.builtin.log10 [[R]] : f64
  dummyB($log10(r));
  // CHECK:  moore.builtin.exp [[R]] : f64
  dummyB($exp(r));
  // CHECK:  moore.builtin.sqrt [[R]] : f64
  dummyB($sqrt(r));
  // CHECK:  moore.builtin.floor [[R]] : f64
  dummyB($floor(r));
  // CHECK:  moore.builtin.ceil [[R]] : f64
  dummyB($ceil(r));
  // CHECK:  moore.builtin.sin [[R]] : f64
  dummyB($sin(r));
  // CHECK:  moore.builtin.cos [[R]] : f64
  dummyB($cos(r));
  // CHECK:  moore.builtin.tan [[R]] : f64
  dummyB($tan(r));
  // CHECK:  moore.builtin.asin [[R]] : f64
  dummyB($asin(r));
  // CHECK:  moore.builtin.acos [[R]] : f64
  dummyB($acos(r));
  // CHECK:  moore.builtin.atan [[R]] : f64
  dummyB($atan(r));
  // CHECK:  moore.builtin.sinh [[R]] : f64
  dummyB($sinh(r));
  // CHECK:  moore.builtin.cosh [[R]] : f64
  dummyB($cosh(r));
  // CHECK:  moore.builtin.tanh [[R]] : f64
  dummyB($tanh(r));
  // CHECK:  moore.builtin.asinh [[R]] : f64
  dummyB($asinh(r));
  // CHECK:  moore.builtin.acosh [[R]] : f64
  dummyB($acosh(r));
  // CHECK:  moore.builtin.atanh [[R]] : f64
  dummyB($atanh(r));

endfunction

// CHECK-LABEL: func.func private @RandomBuiltins(
// CHECK-SAME: [[X:%.+]]: !moore.i32
function RandomBuiltins(int x);
  // CHECK: [[URAND0:%.+]] = moore.builtin.urandom
  // CHECK-NEXT: call @dummyA([[URAND0]]) : (!moore.i32) -> ()
  dummyA($urandom());
  // CHECK: [[URAND1:%.+]] = moore.builtin.urandom [[X]]
  // CHECK-NEXT: call @dummyA([[URAND1]]) : (!moore.i32) -> ()
  dummyA($urandom(x));
  // CHECK: [[RAND0:%.+]] = moore.builtin.random
  // CHECK-NEXT: call @dummyA([[RAND0]]) : (!moore.i32) -> ()
  dummyA($random());
  // CHECK: [[RAND1:%.+]] = moore.builtin.random [[X]]
  // CHECK-NEXT: call @dummyA([[RAND1]]) : (!moore.i32) -> ()
  dummyA($random(x));
endfunction

// CHECK-LABEL: func.func private @TimeBuiltins(
function TimeBuiltins();
  // CHECK: [[TIME:%.+]] = moore.builtin.time
  // CHECK-NEXT: [[TIMETOLOGIC:%.+]] = moore.time_to_logic [[TIME]]
  dummyA($time());
  // CHECK: [[STIME:%.+]] = moore.builtin.time
  dummyA($stime());
  // CHECK: [[REALTIME:%.+]] = moore.builtin.time
  // TODO: There is no int-to-real conversion yet; change this to dummyB once int-to-real works!
  dummyA($realtime());
endfunction

// CHECK-LABEL: func.func private @ConversionBuiltins(
// CHECK-SAME: [[SINT:%.+]]: !moore.i32
// CHECK-SAME: [[LINT:%.+]]: !moore.i64
// CHECK-SAME: [[SR:%.+]]: !moore.f32
// CHECK-SAME: [[R:%.+]]: !moore.f64
function void ConversionBuiltins(int shortint_in, longint longint_in,
                                 shortreal shortreal_in, real real_in);
  // CHECK: [[B2SR:%.+]] = moore.builtin.bitstoshortreal [[SINT]] : i32
  dummyC($bitstoshortreal(shortint_in));
  // CHECK: [[B2R:%.+]] = moore.builtin.bitstoreal [[LINT]] : i64
  dummyB($bitstoreal(longint_in));
  // CHECK: [[R2B:%.+]] = moore.builtin.realtobits [[R]]
  dummyA($realtobits(real_in));
  // CHECK: [[SR2B:%.+]] = moore.builtin.shortrealtobits [[SR]]
  dummyA($shortrealtobits(shortreal_in));
endfunction

// CHECK-LABEL: func.func private @SeverityBuiltinEdgeCase(
// CHECK-SAME: [[STR:%.+]]: !moore.string
function void SeverityBuiltinEdgeCase(string testStr);
  // CHECK: [[CONST:%.+]] = moore.constant 1234 : i32
  // CHECK-NEXT: [[INTVAR:%.+]] = moore.variable [[CONST]] : <i32>
  int val = 1234;
  // CHECK-NEXT: [[INTVAL1:%.+]] = moore.read [[INTVAR]] : <i32>
  // CHECK-NEXT: [[FMTINT1:%.+]] = moore.fmt.int binary [[INTVAL1]], width 32, align right, pad zero : i32
  // CHECK-NEXT: [[LINE1:%.+]] = moore.fmt.literal "\0A"
  // CHECK-NEXT: [[CONCAT1:%.+]] = moore.fmt.concat ([[FMTINT1]], [[LINE1]])
  // CHECK-NEXT: moore.builtin.display [[CONCAT1]]
  $displayb(val);
  // CHECK-NEXT: [[INTVAL2:%.+]] = moore.read [[INTVAR]] : <i32>
  // CHECK-NEXT: [[FMTINT2:%.+]] = moore.fmt.int octal [[INTVAL2]], width 11, align right, pad zero : i32
  // CHECK-NEXT: [[LINE2:%.+]] = moore.fmt.literal "\0A"
  // CHECK-NEXT: [[CONCAT2:%.+]] = moore.fmt.concat ([[FMTINT2]], [[LINE2]])
  // CHECK-NEXT: moore.builtin.display [[CONCAT2]]
  $displayo(val);
  // CHECK-NEXT: [[INTVAL3:%.+]] = moore.read [[INTVAR]] : <i32>
  // CHECK-NEXT: [[FMTINT3:%.+]] = moore.fmt.int hex_lower [[INTVAL3]], width 8, align right, pad zero : i32
  // CHECK-NEXT: [[LINE3:%.+]] = moore.fmt.literal "\0A"
  // CHECK-NEXT: [[CONCAT3:%.+]] = moore.fmt.concat ([[FMTINT3]], [[LINE3]])
  // CHECK-NEXT: moore.builtin.display [[CONCAT3]]
  $displayh(val);
  // CHECK: [[FMTSTR:%.+]] = moore.fmt.string [[STR]]
  // CHECK-NEXT: [[FMTLIT:%.+]] = moore.fmt.literal " 23"
  // CHECK-NEXT: [[FMTCONCAT:%.+]] = moore.fmt.concat ([[FMTSTR]], [[FMTLIT]])
  // CHECK-NEXT: moore.builtin.severity fatal [[FMTCONCAT]]
  // CHECK: moore.unreachable
  $fatal(1, $sformatf("%s 23", testStr));
endfunction

// CHECK-LABEL: moore.module @SampleValueBuiltins(
// CHECK-SAME: in [[CLK:%.+]] : !moore.l1
module SampleValueBuiltins #() (
    input clk_i
);
  // CHECK: [[CLKWIRE:%.+]] = moore.net name "clk_i" wire : <l1>
  // CHECK: moore.procedure always {
  // CHECK-NEXT: [[C:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CB:%.+]] = moore.to_builtin_bool [[C]] : l1
  // CHECK-NEXT: [[C2:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CURRENT:%.+]] = moore.to_builtin_bool [[C2]] : l1
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[CURRENT]], 1 : i1
  // CHECK-NEXT: [[NOTPAST:%.+]] = ltl.not [[PAST]] : !ltl.sequence
  // CHECK-NEXT: [[NOTPASTANDCURRENT:%.+]] = ltl.and [[CURRENT]], [[NOTPAST]] : i1, !ltl.property
  rising_clk: assert property (@(posedge clk_i) clk_i |=> $rose(clk_i));
  // CHECK: moore.procedure always {
  // CHECK-NEXT: [[C:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CB:%.+]] = moore.to_builtin_bool [[C]] : l1
  // CHECK-NEXT: [[C2:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CURRENT:%.+]] = moore.to_builtin_bool [[C2]] : l1
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[CURRENT]], 1 : i1
  // CHECK-NEXT: [[NOTCURRENT:%.+]] = ltl.not [[CURRENT]] : i1
  // CHECK-NEXT: [[PASTANDNOTCURRENT:%.+]] = ltl.and [[NOTCURRENT]], [[PAST]] : !ltl.property, !ltl.sequence
  falling_clk: assert property (@(posedge clk_i) clk_i |=> $fell(clk_i));
  // CHECK: moore.procedure always {
  // CHECK-NEXT: [[C:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CB:%.+]] = moore.to_builtin_bool [[C]] : l1
  // CHECK-NEXT: [[C2:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CURRENT:%.+]] = moore.to_builtin_bool [[C2]] : l1
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[CURRENT]], 1 : i1
  // CHECK-NEXT: [[NOTPAST:%.+]] = ltl.not [[PAST]] : !ltl.sequence
  // CHECK-NEXT: [[NOTCURRENT:%.+]] = ltl.not [[CURRENT]] : i1
  // CHECK-NEXT: [[PASTANDCURRENT:%.+]] = ltl.and [[CURRENT]], [[PAST]] : i1, !ltl.sequence
  // CHECK-NEXT: [[NOTPASTANDNOTCURRENT:%.+]] = ltl.and [[NOTCURRENT]], [[NOTPAST]] : !ltl.property, !ltl.property
  // CHECK-NEXT: [[STABLE:%.+]] = ltl.or [[PASTANDCURRENT]], [[NOTPASTANDNOTCURRENT]] : !ltl.sequence, !ltl.property
  stable_clk: assert property (@(posedge clk_i) clk_i |=> $stable(clk_i));

endmodule

// CHECK-LABEL: func.func private @StringBuiltins(
// CHECK-SAME: [[STR:%.+]]: !moore.string
// CHECK-SAME: [[INT:%.+]]: !moore.i32
function void StringBuiltins(string string_in, int int_in);
  // CHECK: [[LEN:%.+]] = moore.string.len [[STR]]
  dummyA(string_in.len());
  // CHECK: [[LEN:%.+]] = moore.string.toupper [[STR]]
  dummyD(string_in.toupper());
  // CHECK: [[LEN:%.+]] = moore.string.tolower [[STR]]
  dummyD(string_in.tolower());
  // CHECK: [[CHAR:%.+]] = moore.string.getc [[STR]]{{\[}}[[INT]]]
  dummyE(string_in.getc(int_in));
endfunction
