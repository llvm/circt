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

  // CHECK: moore.fmt.int binary [[X]], align right, pad zero : i32
  $write("%b", x);
  // CHECK: moore.fmt.int binary [[X]], align right, pad zero : i32
  $write("%B", x);
  // CHECK: moore.fmt.int binary [[X]], align right, pad zero width 0 : i32
  $write("%0b", x);
  // CHECK: moore.fmt.int binary [[X]], align right, pad zero width 42 : i32
  $write("%42b", x);
  // CHECK: moore.fmt.int binary [[X]], align left, pad zero width 42 : i32
  $write("%-42b", x);

  // CHECK: moore.fmt.int octal [[X]], align right, pad zero : i32
  $write("%o", x);
  // CHECK: moore.fmt.int octal [[X]], align right, pad zero : i32
  $write("%O", x);
  // CHECK: moore.fmt.int octal [[X]], align right, pad zero width 0 : i32
  $write("%0o", x);
  // CHECK: moore.fmt.int octal [[X]], align right, pad zero width 19 : i32
  $write("%19o", x);
  // CHECK: moore.fmt.int octal [[X]], align left, pad zero width 19 : i32
  $write("%-19o", x);

  // CHECK: moore.fmt.real float [[R]], align right : f64
  $write("%f", r);
  // CHECK: moore.fmt.real exponential [[R]], align right : f64
  $write("%e", r);
  // CHECK: moore.fmt.real general [[R]], align right : f64
  $write("%g", r);
  // CHECK: moore.fmt.real float [[R]], align left fracDigits 1 : f64
  $write("%-.f", r);
  // CHECK: moore.fmt.real exponential [[R]], align right fieldWidth 10 : f64
  $write("%10e", r);
  // CHECK: moore.fmt.real general [[R]], align right fracDigits 5 : f64
  $write("%0.5g", r);
  // CHECK: moore.fmt.real float [[R]], align left fieldWidth 20 fracDigits 5 : f64
  $write("%-20.5f", r);
  // CHECK: moore.fmt.real exponential [[R]], align right fieldWidth 10 fracDigits 1 : f64
  $write("%10.e", r);
  // CHECK: moore.fmt.real general [[R]], align right fieldWidth 9 fracDigits 8 : f64
  $write("%9.8g", r);

  // CHECK: [[XR:%.+]] = moore.sint_to_real [[X]] : i32 -> f64
  // CHECK: [[TMP:%.+]] = moore.fmt.real float [[XR]]
  // CHECK: moore.builtin.display [[TMP]]
  $write("%f", x);

  // CHECK: moore.fmt.int decimal [[X]], align right, pad space signed : i32
  $write("%d", x);
  // CHECK: moore.fmt.int decimal [[X]], align right, pad space signed : i32
  $write("%D", x);
  // CHECK: moore.fmt.int decimal [[X]], align right, pad space width 0 signed : i32
  $write("%0d", x);
  // CHECK: moore.fmt.int decimal [[X]], align right, pad space width 19 signed : i32
  $write("%19d", x);
  // CHECK: moore.fmt.int decimal [[X]], align left, pad space width 19 signed : i32
  $write("%-19d", x);

  // CHECK: moore.fmt.int hex_lower [[X]], align right, pad zero : i32
  $write("%h", x);
  // CHECK: moore.fmt.int hex_lower [[X]], align right, pad zero : i32
  $write("%x", x);
  // CHECK: moore.fmt.int hex_upper [[X]], align right, pad zero : i32
  $write("%H", x);
  // CHECK: moore.fmt.int hex_upper [[X]], align right, pad zero : i32
  $write("%X", x);
  // CHECK: moore.fmt.int hex_lower [[X]], align right, pad zero width 0 : i32
  $write("%0h", x);
  // CHECK: moore.fmt.int hex_lower [[X]], align right, pad zero width 19 : i32
  $write("%19h", x);
  // CHECK: moore.fmt.int hex_lower [[X]], align right, pad zero width 19 : i32
  $write("%019h", x);
  // CHECK: moore.fmt.int hex_lower [[X]], align left, pad zero width 19 : i32
  $write("%-19h", x);
  // CHECK: moore.fmt.int hex_lower [[X]], align left, pad zero width 19 : i32
  $write("%-019h", x);

  // CHECK: [[TMP:%.+]] = moore.fmt.int decimal [[X]], align right, pad space signed : i32
  // CHECK: moore.builtin.display [[TMP]]
  $write(x);
  // CHECK: [[TMP:%.+]] = moore.fmt.int binary [[X]], align right, pad zero : i32
  // CHECK: moore.builtin.display [[TMP]]
  $writeb(x);
  // CHECK: [[TMP:%.+]] = moore.fmt.int octal [[X]], align right, pad zero : i32
  // CHECK: moore.builtin.display [[TMP]]
  $writeo(x);
  // CHECK: [[TMP:%.+]] = moore.fmt.int hex_lower [[X]], align right, pad zero : i32
  // CHECK: moore.builtin.display [[TMP]]
  $writeh(x);

  // CHECK: [[TMP1:%.+]] = moore.fmt.int decimal [[X]], align right, pad space signed : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $display(x);
  // CHECK: [[TMP1:%.+]] = moore.fmt.int binary [[X]], align right, pad zero : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $displayb(x);
  // CHECK: [[TMP1:%.+]] = moore.fmt.int octal [[X]], align right, pad zero : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $displayo(x);
  // CHECK: [[TMP1:%.+]] = moore.fmt.int hex_lower [[X]], align right, pad zero : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $displayh(x);

  // CHECK: [[TMP1:%.+]] = moore.fmt.real float [[R]]
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $display("%f", r);

  // IEEE 1800-2017 § 21.2.1.6 "Hierarchical name format"
  // CHECK: [[TMP1:%.+]] = moore.fmt.hier_path
  // CHECK-NOT: escaped
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $display("%m");
  // CHECK: [[TMP1:%.+]] = moore.fmt.hier_path escaped
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $display("%M");

  // IEEE 1800-2017 § 33.7 "Displaying library binding information"
  // CHECK: [[TMP1:%.+]] = moore.fmt.literal ""
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $display("%l");
  // CHECK: [[TMP1:%.+]] = moore.fmt.literal ""
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $display("%L");

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
  // A leading non-string literal is a displayed argument, and any string
  // literal among the arguments acts as a format string, exactly like
  // `$display` (IEEE 1800-2023 § 20.10).
  // CHECK: [[VERB:%.+]] = moore.fmt.int
  // CHECK: [[MSG:%.+]] = moore.fmt.int
  // CHECK: [[CAT:%.+]] = moore.fmt.concat ([[VERB]], [[MSG]])
  // CHECK: moore.builtin.severity warning [[CAT]]
  $warning(1, "%d", x);
  // CHECK: [[ARG0:%.+]] = moore.fmt.int
  // CHECK: [[LIT:%.+]] = moore.fmt.literal " x="
  // CHECK: [[ARG1:%.+]] = moore.fmt.int
  // CHECK: [[CAT:%.+]] = moore.fmt.concat ([[ARG0]], [[LIT]], [[ARG1]])
  // CHECK: moore.builtin.severity error [[CAT]]
  $error(x, " x=", x);
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

module LibraryBindingBuiltin;
  initial begin
     // IEEE 1800-2017 § 33.7 "Displaying library binding information"
    // CHECK: [[TMP1:%.+]] = moore.fmt.literal "work.LibraryBindingBuiltin"
    // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
    // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
    // CHECK: moore.builtin.display [[TMP3]]
    $display("%l");
    // CHECK: [[TMP1:%.+]] = moore.fmt.literal "work.LibraryBindingBuiltin"
    // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
    // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
    // CHECK: moore.builtin.display [[TMP3]]
    $display("%L");
  end
endmodule

// CHECK-LABEL: func.func private @FormatCharSpecifier(
function void FormatCharSpecifier();
  byte ch = 88;
  int ch_int = 235800858;
  // CHECK: moore.fmt.char {{%.+}} : i8
  $display("%c", ch);
  // CHECK: moore.fmt.char {{%.+}} : i8
  $display("%C", ch);
  // CHECK: moore.fmt.char {{%.+}} : i32
  $display("%c", ch_int);
  // CHECK: moore.fmt.char {{%.+}} : i32
  $display("%C", ch_int);
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

// IEEE 1800-2023 $ 18.3 "Random number system functions and methods"
// CHECK-LABEL: func.func private @RandomBuiltins(
// CHECK-SAME: [[X:%.+]]: !moore.i32
function RandomBuiltins(int x);
  // CHECK: [[XVAR:%.+]] = moore.variable [[X]] : <i32>
  // All random system functions map to moore.builtin.urandom_range with
  // (minval, maxval, optional seed ref).

  // CHECK: [[ZERO1:%.+]] = moore.constant 0 : i32
  // CHECK: [[MAX1:%.+]] = moore.constant -1 : i32
  // CHECK: [[URAND0:%.+]] = moore.builtin.urandom_range [[ZERO1]], [[MAX1]]
  // CHECK-NEXT: call @dummyA([[URAND0]]) : (!moore.i32) -> ()
  dummyA($urandom());

  // CHECK: [[ZERO2:%.+]] = moore.constant 0 : i32
  // CHECK: [[MAX2:%.+]] = moore.constant -1 : i32
  // CHECK: [[URAND1:%.+]] = moore.builtin.urandom_range [[ZERO2]], [[MAX2]], [[XVAR]]
  // CHECK-NEXT: call @dummyA([[URAND1]]) : (!moore.i32) -> ()
  dummyA($urandom(x));

  // CHECK: [[ZERO3:%.+]] = moore.constant 0 : i32
  // CHECK: [[MAX3:%.+]] = moore.constant -1 : i32
  // CHECK: [[RAND0:%.+]] = moore.builtin.urandom_range [[ZERO3]], [[MAX3]]
  // CHECK-NEXT: call @dummyA([[RAND0]]) : (!moore.i32) -> ()
  dummyA($random());

  // CHECK: [[ZERO4:%.+]] = moore.constant 0 : i32
  // CHECK: [[MAX4:%.+]] = moore.constant -1 : i32
  // CHECK: [[RAND1:%.+]] = moore.builtin.urandom_range [[ZERO4]], [[MAX4]], [[XVAR]]
  // CHECK-NEXT: call @dummyA([[RAND1]]) : (!moore.i32) -> ()
  dummyA($random(x));

  // CHECK: [[XVAL1:%.+]] = moore.read [[XVAR]]
  // CHECK: [[ZERO5:%.+]] = moore.constant 0 : i32
  // CHECK: [[URANDRANGE1:%.+]] = moore.builtin.urandom_range [[ZERO5]], [[XVAL1]]
  // CHECK-NEXT: call @dummyA([[URANDRANGE1]]) : (!moore.i32) -> ()
  dummyA($urandom_range(x));

  // CHECK: [[XVAL2:%.+]] = moore.read [[XVAR]]
  // CHECK: [[XVAL3:%.+]] = moore.read [[XVAR]]
  // CHECK: [[URANDRANGE2:%.+]] = moore.builtin.urandom_range [[XVAL3]], [[XVAL2]]
  // CHECK-NEXT: call @dummyA([[URANDRANGE2]]) : (!moore.i32) -> ()
  dummyA($urandom_range(x, x));
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
  // CHECK-NEXT: [[FMTINT1:%.+]] = moore.fmt.int binary [[INTVAL1]], align right, pad zero : i32
  // CHECK-NEXT: [[LINE1:%.+]] = moore.fmt.literal "\0A"
  // CHECK-NEXT: [[CONCAT1:%.+]] = moore.fmt.concat ([[FMTINT1]], [[LINE1]])
  // CHECK-NEXT: moore.builtin.display [[CONCAT1]]
  $displayb(val);
  // CHECK-NEXT: [[INTVAL2:%.+]] = moore.read [[INTVAR]] : <i32>
  // CHECK-NEXT: [[FMTINT2:%.+]] = moore.fmt.int octal [[INTVAL2]], align right, pad zero : i32
  // CHECK-NEXT: [[LINE2:%.+]] = moore.fmt.literal "\0A"
  // CHECK-NEXT: [[CONCAT2:%.+]] = moore.fmt.concat ([[FMTINT2]], [[LINE2]])
  // CHECK-NEXT: moore.builtin.display [[CONCAT2]]
  $displayo(val);
  // CHECK-NEXT: [[INTVAL3:%.+]] = moore.read [[INTVAR]] : <i32>
  // CHECK-NEXT: [[FMTINT3:%.+]] = moore.fmt.int hex_lower [[INTVAL3]], align right, pad zero : i32
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

// IEEE 1800-2023 § 16.9.3 "Sampled value functions"
// CHECK-LABEL: moore.module @SampleValueBuiltins(
// CHECK-SAME: in [[CLK:%.+]] : !moore.l1
module SampleValueBuiltins #() (
    input clk_i,
    input [7:0] data_i,
    input bit [7:0] data_bit_i
);
  // CHECK: [[CLKWIRE:%.+]] = moore.net name "clk_i" wire : <l1>
  // CHECK: [[DATAWIRE:%.+]] = moore.net name "data_i" wire : <l8>
  // CHECK: [[DATABITWIRE:%.+]] = moore.net name "data_bit_i" wire : <i8>
  // CHECK: moore.procedure always {
  // CHECK-NEXT: [[C:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[C_INT:%.+]] = moore.logic_to_int [[C]] : l1
  // CHECK-NEXT: [[CB:%.+]] = moore.to_builtin_int [[C_INT]] : i1
  // CHECK-NEXT: [[CLK:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[C2:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[C2_INT:%.+]] = moore.logic_to_int [[C2]] : l1
  // CHECK-NEXT: [[CURRENT:%.+]] = moore.to_builtin_int [[C2_INT]] : i1
  // CHECK-NEXT: [[SAMPLED:%.+]] = ltl.sampled [[CURRENT]] : i1
  sampled_clk: assert property (@(posedge clk_i) clk_i |=> $sampled(clk_i));
  // CHECK: moore.procedure always {
  // CHECK-NEXT: [[C:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[C_INT:%.+]] = moore.logic_to_int [[C]] : l1
  // CHECK-NEXT: [[CB:%.+]] = moore.to_builtin_int [[C_INT]] : i1
  // CHECK-NEXT: [[CLK:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[C2:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[C2_INT:%.+]] = moore.logic_to_int [[C2]] : l1
  // CHECK-NEXT: [[CURRENT:%.+]] = moore.to_builtin_int [[C2_INT]] : i1
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[CURRENT]], 1 clk [[CLK_I1]] : i1
  // CHECK-NEXT: [[ROSE:%.+]] = comb.icmp ult [[PAST]], [[CURRENT]] : i1
  rising_clk: assert property (@(posedge clk_i) clk_i |=> $rose(clk_i));
  // Check that the output of rose can be used by non-LTL ops
  // CHECK: moore.procedure always {
  // CHECK-NEXT: [[C1:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[C2:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[C2_INT:%.+]] = moore.logic_to_int [[C2]] : l1
  // CHECK-NEXT: [[CURRENT:%.+]] = moore.to_builtin_int [[C2_INT]] : i1
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[CURRENT]], 1 clk [[CLK_I1]] : i1
  // CHECK-NEXT: [[ROSE:%.+]] = comb.icmp ult [[PAST]], [[CURRENT]] : i1
  // CHECK-NEXT: [[ROSE_INT:%.+]] = moore.from_builtin_int [[ROSE]] : i1
  // CHECK-NEXT: [[ROSE_LOGIC:%.+]] = moore.int_to_logic [[ROSE_INT]] : i1
  // CHECK-NEXT: [[EQ:%.+]] = moore.eq [[C1]], [[ROSE_LOGIC]] : l1 -> l1
  rose_eq: assert property (@(posedge clk_i) clk_i == $rose(clk_i));
  // CHECK: moore.procedure always {
  // CHECK-NEXT: [[C:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[C_INT:%.+]] = moore.logic_to_int [[C]] : l1
  // CHECK-NEXT: [[CB:%.+]] = moore.to_builtin_int [[C_INT]] : i1
  // CHECK-NEXT: [[CLK:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[C2:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[C2_INT:%.+]] = moore.logic_to_int [[C2]] : l1
  // CHECK-NEXT: [[CURRENT:%.+]] = moore.to_builtin_int [[C2_INT]] : i1
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[CURRENT]], 1 clk [[CLK_I1]] : i1
  // CHECK-NEXT: [[FELL:%.+]] = comb.icmp ugt [[PAST]], [[CURRENT]] : i1
  falling_clk: assert property (@(posedge clk_i) clk_i |=> $fell(clk_i));
  // Check that the output of fell can be used by non-LTL ops
  // CHECK: moore.procedure always {
  // CHECK-NEXT: [[C1:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[C2:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[C2_INT:%.+]] = moore.logic_to_int [[C2]] : l1
  // CHECK-NEXT: [[CURRENT:%.+]] = moore.to_builtin_int [[C2_INT]] : i1
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[CURRENT]], 1 clk [[CLK_I1]] : i1
  // CHECK-NEXT: [[FELL:%.+]] = comb.icmp ugt [[PAST]], [[CURRENT]] : i1
  // CHECK-NEXT: [[FELL_INT:%.+]] = moore.from_builtin_int [[FELL]] : i1
  // CHECK-NEXT: [[FELL_LOGIC:%.+]] = moore.int_to_logic [[FELL_INT]] : i1
  // CHECK-NEXT: [[EQ:%.+]] = moore.eq [[C1]], [[FELL_LOGIC]] : l1 -> l1
  fell_eq: assert property (@(posedge clk_i) clk_i == $fell(clk_i));
  // CHECK: moore.procedure always {
  // CHECK-NEXT: [[C:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[C_INT:%.+]] = moore.logic_to_int [[C]] : l1
  // CHECK-NEXT: [[CB:%.+]] = moore.to_builtin_int [[C_INT]] : i1
  // CHECK-NEXT: [[CLK:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[C2:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[C2_INT:%.+]] = moore.logic_to_int [[C2]] : l1
  // CHECK-NEXT: [[CURRENT:%.+]] = moore.to_builtin_int [[C2_INT]] : i1
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[CURRENT]], 1 clk [[CLK_I1]] : i1
  // CHECK-NEXT: [[STABLE:%.+]] = comb.icmp eq [[PAST]], [[CURRENT]] : i1
  stable_clk: assert property (@(posedge clk_i) clk_i |=> $stable(clk_i));
  // Check that the output of stable can be used by non-LTL ops
  // CHECK: moore.procedure always {
  // CHECK-NEXT: [[C1:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[C2:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[C2_INT:%.+]] = moore.logic_to_int [[C2]] : l1
  // CHECK-NEXT: [[CURRENT:%.+]] = moore.to_builtin_int [[C2_INT]] : i1
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[CURRENT]], 1 clk [[CLK_I1]] : i1
  // CHECK-NEXT: [[STABLE:%.+]] = comb.icmp eq [[PAST]], [[CURRENT]] : i1
  // CHECK-NEXT: [[STABLE_INT:%.+]] = moore.from_builtin_int [[STABLE]] : i1
  // CHECK-NEXT: [[STABLE_LOGIC:%.+]] = moore.int_to_logic [[STABLE_INT]] : i1
  // CHECK-NEXT: [[EQ:%.+]] = moore.eq [[C1]], [[STABLE_LOGIC]] : l1 -> l1
  stable_eq: assert property (@(posedge clk_i) clk_i == $stable(clk_i));
  // CHECK: moore.procedure always {
  // CHECK-NEXT: [[C:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[C_INT:%.+]] = moore.logic_to_int [[C]] : l1
  // CHECK-NEXT: [[CB:%.+]] = moore.to_builtin_int [[C_INT]] : i1
  // CHECK-NEXT: [[CLK:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[C2:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[C2_INT:%.+]] = moore.logic_to_int [[C2]] : l1
  // CHECK-NEXT: [[CURRENT:%.+]] = moore.to_builtin_int [[C2_INT]] : i1
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[CURRENT]], 1 clk [[CLK_I1]] : i1
  // CHECK-NEXT: [[CHANGED:%.+]] = comb.icmp ne [[PAST]], [[CURRENT]] : i1
  changed_clk: assert property (@(posedge clk_i) clk_i |=> $changed(clk_i));
  // Check that the output of changed can be used by non-LTL ops
  // CHECK: moore.procedure always {
  // CHECK-NEXT: [[C1:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[C2:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[C2_INT:%.+]] = moore.logic_to_int [[C2]] : l1
  // CHECK-NEXT: [[CURRENT:%.+]] = moore.to_builtin_int [[C2_INT]] : i1
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[CURRENT]], 1 clk [[CLK_I1]] : i1
  // CHECK-NEXT: [[CHANGED:%.+]] = comb.icmp ne [[PAST]], [[CURRENT]] : i1
  // CHECK-NEXT: [[CHANGED_INT:%.+]] = moore.from_builtin_int [[CHANGED]] : i1
  // CHECK-NEXT: [[CHANGED_LOGIC:%.+]] = moore.int_to_logic [[CHANGED_INT]] : i1
  // CHECK-NEXT: [[EQ:%.+]] = moore.eq [[C1]], [[CHANGED_LOGIC]] : l1 -> l1
  changed_eq: assert property (@(posedge clk_i) clk_i == $changed(clk_i));
  // CHECK: moore.procedure always {
  // CHECK-NEXT: [[C:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[C_INT:%.+]] = moore.logic_to_int [[C]] : l1
  // CHECK-NEXT: [[CB:%.+]] = moore.to_builtin_int [[C_INT]] : i1
  // CHECK-NEXT: [[CLK:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[C2:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[C2_INT:%.+]] = moore.logic_to_int [[C2]] : l1
  // CHECK-NEXT: [[CURRENT:%.+]] = moore.to_builtin_int [[C2_INT]] : i1
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[CURRENT]], 1 clk [[CLK_I1]] : i1
  past_clk: assert property (@(posedge clk_i) clk_i |=> $past(clk_i));
  // Check that the output of past can be used by non-LTL ops
  // CHECK: moore.procedure always {
  // CHECK-NEXT: [[C1:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[C2:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[C2_INT:%.+]] = moore.logic_to_int [[C2]] : l1
  // CHECK-NEXT: [[CB:%.+]] = moore.to_builtin_int [[C2_INT]] : i1
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[CB]], 1 clk [[CLK_I1]] : i1
  // CHECK-NEXT: [[PAST_INT:%.+]] = moore.from_builtin_int [[PAST]] : i1
  // CHECK-NEXT: [[PAST_LOGIC:%.+]] = moore.int_to_logic [[PAST_INT]] : i1
  // CHECK-NEXT: [[EQ:%.+]] = moore.eq [[C1]], [[PAST_LOGIC]] : l1 -> l1
  past_eq: assert property (@(posedge clk_i) clk_i == $past(clk_i));
  // Test $past on wider bitvectors
  // CHECK: moore.procedure always {
  // CHECK-NEXT: [[D1:%.+]] = moore.read [[DATAWIRE]] : <l8>
  // CHECK-NEXT: [[CLK:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[D2:%.+]] = moore.read [[DATAWIRE]] : <l8>
  // CHECK-NEXT: [[D2_INT:%.+]] = moore.logic_to_int [[D2]] : l8
  // CHECK-NEXT: [[DB:%.+]] = moore.to_builtin_int [[D2_INT]] : i8
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[DB]], 1 clk [[CLK_I1]] : i8
  // CHECK-NEXT: [[PAST_INT:%.+]] = moore.from_builtin_int [[PAST]] : i8
  // CHECK-NEXT: [[PAST_LOGIC:%.+]] = moore.int_to_logic [[PAST_INT]] : i8
  // CHECK-NEXT: [[EQ:%.+]] = moore.eq [[D1]], [[PAST_LOGIC]] : l8 -> l1
  past_data: assert property (@(posedge clk_i) data_i == $past(data_i));

  // Test $past in a process
  // CHECK: moore.procedure always {
  // CHECK-NEXT: moore.wait_event {
  // CHECK-NEXT:   [[CLKEDGE:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT:   moore.detect_event posedge [[CLKEDGE]] : l1
  // CHECK-NEXT: }
  // CHECK-NEXT: [[CLK:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[D1:%.+]] = moore.read [[DATAWIRE]] : <l8>
  // CHECK-NEXT: [[D1_INT:%.+]] = moore.logic_to_int [[D1]] : l8
  // CHECK-NEXT: [[DB:%.+]] = moore.to_builtin_int [[D1_INT]] : i8
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[DB]], 1 clk [[CLK_I1]] : i8
  // CHECK-NEXT: [[PAST_INT:%.+]] = moore.from_builtin_int [[PAST]] : i8
  // CHECK-NEXT: [[PAST_LOGIC:%.+]] = moore.int_to_logic [[PAST_INT]] : i8
  // CHECK-NEXT: moore.nonblocking_assign {{%.+}}, [[PAST_LOGIC]] : l8
  logic [7:0] past_data_i;
  always @(posedge clk_i) past_data_i <= $past(data_i);

  // CHECK: moore.procedure always {
  // CHECK: [[D:%.+]] = moore.read [[DATAWIRE]] : <l8>
  // CHECK: [[RED:%.+]] = moore.reduce_xor [[D]] : l8 -> l1
  // CHECK: [[X:%.+]] = moore.constant bX : l1
  // CHECK: [[CEQ:%.+]] = moore.case_eq [[RED]], [[X]] : l1
  // CHECK: [[CEQ_I1:%.+]] = moore.to_builtin_int [[CEQ]] : i1
  // CHECK: ltl.clock [[CEQ_I1]]
  isunknown_data: assert property (@(posedge clk_i) $isunknown(data_i));

  // CHECK: moore.procedure always {
  // CHECK: [[D:%.+]] = moore.read [[DATAWIRE]] : <l8>
  // CHECK: [[RED:%.+]] = moore.reduce_xor [[D]] : l8 -> l1
  // CHECK: [[X:%.+]] = moore.constant bX : l1
  // CHECK: [[ISUNKNOWN:%.+]] = moore.case_eq [[RED]], [[X]] : l1
  // CHECK: [[ISUNKNOWN_I1:%.+]] = moore.to_builtin_int [[ISUNKNOWN]] : i1
  // CHECK: [[D_L2I:%.+]] = moore.logic_to_int [[D]] : l8
  // CHECK: [[DB:%.+]] = moore.to_builtin_int [[D_L2I]] : i8
  // CHECK: [[ONE:%.+]] = hw.constant 1 : i8
  // CHECK: [[SUB:%.+]] = comb.sub [[DB]], [[ONE]] : i8
  // CHECK: [[AND:%.+]] = comb.and [[DB]], [[SUB]] : i8
  // CHECK: [[ZERO:%.+]] = hw.constant 0 : i8
  // CHECK: [[EQ:%.+]] = comb.icmp eq [[AND]], [[ZERO]] : i8
  // CHECK: [[FALSE:%.+]] = hw.constant false
  // CHECK: [[MUX:%.+]] = comb.mux [[ISUNKNOWN_I1]], [[FALSE]], [[EQ]] : i1
  // CHECK: [[RES_INT:%.+]] = moore.from_builtin_int [[MUX]] : i1
  // CHECK: [[RES_LOGIC:%.+]] = moore.int_to_logic [[RES_INT]] : i1
  // CHECK: [[RES_BUILTIN:%.+]] = moore.to_builtin_int [[RES_INT]] : i1
  // CHECK: ltl.clock [[RES_BUILTIN]]
  onehot0_data: assert property (@(posedge clk_i) $onehot0(data_i));

  // CHECK: moore.procedure always {
  // CHECK: [[D:%.+]] = moore.read [[DATAWIRE]] : <l8>
  // CHECK: [[RED:%.+]] = moore.reduce_xor [[D]] : l8 -> l1
  // CHECK: [[X:%.+]] = moore.constant bX : l1
  // CHECK: [[ISUNKNOWN:%.+]] = moore.case_eq [[RED]], [[X]] : l1
  // CHECK: [[ISUNKNOWN_I1:%.+]] = moore.to_builtin_int [[ISUNKNOWN]] : i1
  // CHECK: [[D_L2I:%.+]] = moore.logic_to_int [[D]] : l8
  // CHECK: [[DB:%.+]] = moore.to_builtin_int [[D_L2I]] : i8
  // CHECK: [[ONE:%.+]] = hw.constant 1 : i8
  // CHECK: [[SUB:%.+]] = comb.sub [[DB]], [[ONE]] : i8
  // CHECK: [[AND:%.+]] = comb.and [[DB]], [[SUB]] : i8
  // CHECK: [[ZERO:%.+]] = hw.constant 0 : i8
  // CHECK: [[EQ:%.+]] = comb.icmp eq [[AND]], [[ZERO]] : i8
  // CHECK: [[NE:%.+]] = comb.icmp ne [[DB]], [[ZERO]] : i8
  // CHECK: [[AND2:%.+]] = comb.and [[EQ]], [[NE]] : i1
  // CHECK: [[FALSE:%.+]] = hw.constant false
  // CHECK: [[MUX:%.+]] = comb.mux [[ISUNKNOWN_I1]], [[FALSE]], [[AND2]] : i1
  // CHECK: [[RES_INT:%.+]] = moore.from_builtin_int [[MUX]] : i1
  // CHECK: [[RES_LOGIC:%.+]] = moore.int_to_logic [[RES_INT]] : i1
  // CHECK: [[RES_BUILTIN:%.+]] = moore.to_builtin_int [[RES_INT]] : i1
  // CHECK: ltl.clock [[RES_BUILTIN]]
  onehot_data: assert property (@(posedge clk_i) $onehot(data_i));

  // CHECK: moore.procedure always {
  // CHECK: [[D:%.+]] = moore.read [[DATABITWIRE]] : <i8>
  // CHECK: [[DB:%.+]] = moore.to_builtin_int [[D]] : i8
  // CHECK: [[ONE:%.+]] = hw.constant 1 : i8
  // CHECK: [[SUB:%.+]] = comb.sub [[DB]], [[ONE]] : i8
  // CHECK: [[AND:%.+]] = comb.and [[DB]], [[SUB]] : i8
  // CHECK: [[ZERO:%.+]] = hw.constant 0 : i8
  // CHECK: [[EQ:%.+]] = comb.icmp eq [[AND]], [[ZERO]] : i8
  // CHECK: [[EQ_INT:%.+]] = moore.from_builtin_int [[EQ]] : i1
  // CHECK: [[EQ_BUILTIN:%.+]] = moore.to_builtin_int [[EQ_INT]] : i1
  // CHECK: ltl.clock [[EQ_BUILTIN]]
  onehot0_bit_data: assert property (@(posedge clk_i) $onehot0(data_bit_i));

  // CHECK: moore.procedure always {
  // CHECK: [[D:%.+]] = moore.read [[DATABITWIRE]] : <i8>
  // CHECK: [[DB:%.+]] = moore.to_builtin_int [[D]] : i8
  // CHECK: [[ONE:%.+]] = hw.constant 1 : i8
  // CHECK: [[SUB:%.+]] = comb.sub [[DB]], [[ONE]] : i8
  // CHECK: [[AND:%.+]] = comb.and [[DB]], [[SUB]] : i8
  // CHECK: [[ZERO:%.+]] = hw.constant 0 : i8
  // CHECK: [[EQ:%.+]] = comb.icmp eq [[AND]], [[ZERO]] : i8
  // CHECK: [[NE:%.+]] = comb.icmp ne [[DB]], [[ZERO]] : i8
  // CHECK: [[AND2:%.+]] = comb.and [[EQ]], [[NE]] : i1
  // CHECK: [[RES_INT:%.+]] = moore.from_builtin_int [[AND2]] : i1
  // CHECK: [[RES_BUILTIN:%.+]] = moore.to_builtin_int [[RES_INT]] : i1
  // CHECK: ltl.clock [[RES_BUILTIN]]
  onehot_bit_data: assert property (@(posedge clk_i) $onehot(data_bit_i));

  // CHECK: moore.procedure always {
  // CHECK: [[D:%.+]] = moore.read [[DATAWIRE]] : <l8>
  // CHECK: [[D_L2I:%.+]] = moore.logic_to_int [[D]] : l8
  // CHECK: [[DB:%.+]] = moore.to_builtin_int [[D_L2I]] : i8
  // CHECK: [[Z3:%.+]] = hw.constant 0 : i3
  // CHECK: [[B0:%.+]] = comb.extract [[DB]] from 0 : (i8) -> i1
  // CHECK: [[EXT0:%.+]] = comb.concat [[Z3]], [[B0]] : i3, i1
  // CHECK: [[B1:%.+]] = comb.extract [[DB]] from 1 : (i8) -> i1
  // CHECK: [[EXT1:%.+]] = comb.concat [[Z3]], [[B1]] : i3, i1
  // CHECK: [[RES_INT:%.+]] = moore.from_builtin_int {{%.+}} : i4
  // CHECK: [[SEXT:%.+]] = moore.zext [[RES_INT]] : i4 -> i32
  // CHECK: [[ZERO:%.+]] = moore.constant 0 : i32
  // CHECK: [[EQ:%.+]] = moore.eq [[SEXT]], [[ZERO]] : i32 -> i1
  countones_data:
    assert property (@(posedge clk_i) $countones(data_i) == 0);

  // CHECK: moore.procedure always {
  // CHECK: [[D:%.+]] = moore.read [[DATABITWIRE]] : <i8>
  // CHECK: [[DB:%.+]] = moore.to_builtin_int [[D]] : i8
  // CHECK: [[Z3:%.+]] = hw.constant 0 : i3
  // CHECK: [[B0:%.+]] = comb.extract [[DB]] from 0 : (i8) -> i1
  // CHECK: [[EXT0:%.+]] = comb.concat [[Z3]], [[B0]] : i3, i1
  // CHECK: [[B1:%.+]] = comb.extract [[DB]] from 1 : (i8) -> i1
  // CHECK: [[EXT1:%.+]] = comb.concat [[Z3]], [[B1]] : i3, i1
  // CHECK: comb.add
  // CHECK: [[RES_INT:%.+]] = moore.from_builtin_int {{%.+}} : i4
  // CHECK: [[SEXT:%.+]] = moore.zext [[RES_INT]] : i4 -> i32
  // CHECK: [[ZERO:%.+]] = moore.constant 0 : i32
  // CHECK: [[EQ:%.+]] = moore.eq [[SEXT]], [[ZERO]] : i32 -> i1
  countones_bit_data:
    assert property (@(posedge clk_i) $countones(data_bit_i) == 0);
endmodule

// IEEE 1800-2023 § 16.9.3 "Sampled value functions" with default clocking
// This is in a separate module to SampleValueBuiltins as default clocking
// silently propagates to all potential users in the scope.
// CHECK-LABEL: moore.module @SampleValueBuiltinsDefaultClocking(
// CHECK-SAME: in [[CLK:%.+]] : !moore.l1
module SampleValueBuiltinsDefaultClocking #() (
    input clk_i,
    input clk2_i,
    input [7:0] data_i
);
  // CHECK: [[CLKWIRE:%.+]] = moore.net name "clk_i" wire : <l1>
  // CHECK: [[CLK2WIRE:%.+]] = moore.net name "clk2_i" wire : <l1>
  // CHECK: [[DATAWIRE:%.+]] = moore.net name "data_i" wire : <l8>

  default clocking @(posedge clk_i); endclocking

  // Test default clocking inference for $past in an assertion
  // CHECK: moore.procedure always {
  // CHECK-NEXT: [[D1:%.+]] = moore.read [[DATAWIRE]] : <l8>
  // CHECK-NEXT: [[CLK:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[D2:%.+]] = moore.read [[DATAWIRE]] : <l8>
  // CHECK-NEXT: [[D2_INT:%.+]] = moore.logic_to_int [[D2]] : l8
  // CHECK-NEXT: [[DB:%.+]] = moore.to_builtin_int [[D2_INT]] : i8
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[DB]], 1 clk [[CLK_I1]] : i8
  // CHECK-NEXT: [[PAST_INT:%.+]] = moore.from_builtin_int [[PAST]] : i8
  // CHECK-NEXT: [[PAST_LOGIC:%.+]] = moore.int_to_logic [[PAST_INT]] : i8
  // CHECK-NEXT: [[EQ:%.+]] = moore.eq [[D1]], [[PAST_LOGIC]] : l8 -> l1
  assert_past: assert property (data_i == $past(data_i));

  // Test overriden clock in an assertion
  // CHECK: moore.procedure always {
  // CHECK-NEXT: [[D1:%.+]] = moore.read [[DATAWIRE]] : <l8>
  // CHECK-NEXT: [[CLK:%.+]] = moore.read [[CLK2WIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[D2:%.+]] = moore.read [[DATAWIRE]] : <l8>
  // CHECK-NEXT: [[D2_INT:%.+]] = moore.logic_to_int [[D2]] : l8
  // CHECK-NEXT: [[DB:%.+]] = moore.to_builtin_int [[D2_INT]] : i8
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[DB]], 1 clk [[CLK_I1]] : i8
  // CHECK-NEXT: [[PAST_INT:%.+]] = moore.from_builtin_int [[PAST]] : i8
  // CHECK-NEXT: [[PAST_LOGIC:%.+]] = moore.int_to_logic [[PAST_INT]] : i8
  // CHECK-NEXT: [[EQ:%.+]] = moore.eq [[D1]], [[PAST_LOGIC]] : l8 -> l1
  assert_past_clk2: assert property (@(posedge clk2_i) data_i == $past(data_i));

  // Test default clocking inference for $past in a continuous assignment
  // CHECK: [[CLK:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[D1:%.+]] = moore.read [[DATAWIRE]] : <l8>
  // CHECK-NEXT: [[D1_INT:%.+]] = moore.logic_to_int [[D1]] : l8
  // CHECK-NEXT: [[DB:%.+]] = moore.to_builtin_int [[D1_INT]] : i8
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[DB]], 1 clk [[CLK_I1]] : i8
  // CHECK-NEXT: [[PAST_INT:%.+]] = moore.from_builtin_int [[PAST]] : i8
  // CHECK-NEXT: [[PAST_LOGIC:%.+]] = moore.int_to_logic [[PAST_INT]] : i8
  // CHECK-NEXT: moore.assign {{%.+}}, [[PAST_LOGIC]] : l8
  logic [7:0] assign_past;
  assign assign_past = $past(data_i);

  // Test default clocking inference for $past in a combinational procedure
  // CHECK: moore.procedure always_comb {
  // CHECK-NEXT: [[CLK:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[D1:%.+]] = moore.read [[DATAWIRE]] : <l8>
  // CHECK-NEXT: [[D1_INT:%.+]] = moore.logic_to_int [[D1]] : l8
  // CHECK-NEXT: [[DB:%.+]] = moore.to_builtin_int [[D1_INT]] : i8
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[DB]], 1 clk [[CLK_I1]] : i8
  // CHECK-NEXT: [[PAST_INT:%.+]] = moore.from_builtin_int [[PAST]] : i8
  // CHECK-NEXT: [[PAST_LOGIC:%.+]] = moore.int_to_logic [[PAST_INT]] : i8
  // CHECK-NEXT: moore.blocking_assign {{%.+}}, [[PAST_LOGIC]] : l8
  logic [7:0] comb_assign_past;
  always_comb comb_assign_past = $past(data_i);

  // Test overriden clock for $past in a procedure
  // CHECK: moore.procedure always_ff {
  // CHECK-NEXT: moore.wait_event {
  // CHECK-NEXT:   [[CLKEDGE:%.+]] = moore.read [[CLK2WIRE]] : <l1>
  // CHECK-NEXT:   moore.detect_event posedge [[CLKEDGE]] : l1
  // CHECK-NEXT: }
  // CHECK-NEXT: [[CLK:%.+]] = moore.read [[CLK2WIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[D1:%.+]] = moore.read [[DATAWIRE]] : <l8>
  // CHECK-NEXT: [[D1_INT:%.+]] = moore.logic_to_int [[D1]] : l8
  // CHECK-NEXT: [[DB:%.+]] = moore.to_builtin_int [[D1_INT]] : i8
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[DB]], 1 clk [[CLK_I1]] : i8
  // CHECK-NEXT: [[PAST_INT:%.+]] = moore.from_builtin_int [[PAST]] : i8
  // CHECK-NEXT: [[PAST_LOGIC:%.+]] = moore.int_to_logic [[PAST_INT]] : i8
  // CHECK-NEXT: moore.nonblocking_assign {{%.+}}, [[PAST_LOGIC]] : l8
  logic [7:0] clk2_assign_past;
  always_ff @(posedge clk2_i) clk2_assign_past <= $past(data_i);
endmodule

// IEEE 1800-2023 § 16.9.3 "Sampled value functions" with default clocking
// outside of procedures
// CHECK-LABEL: moore.module @SampleValueBuiltinsDefaultClockingNoProcedure(
// CHECK-SAME: in [[CLK:%.+]] : !moore.l1
module SampleValueBuiltinsDefaultClockingNoProcedure #() (
    input clk_i,
    input [7:0] data_i
);
  // CHECK: [[CLKWIRE:%.+]] = moore.net name "clk_i" wire : <l1>
  // CHECK: [[DATAWIRE:%.+]] = moore.net name "data_i" wire : <l8>
  // CHECK: [[WIRE_PAST:%.+]] = moore.net wire [[WIRE_PAST_VAL:%.+]] : <l8>

  default clocking @(posedge clk_i); endclocking

  // CHECK: [[CLK:%.+]] = moore.read [[CLKWIRE]] : <l1>
  // CHECK-NEXT: [[CLK_INT:%.+]] = moore.logic_to_int [[CLK]] : l1
  // CHECK-NEXT: [[CLK_I1:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
  // CHECK-NEXT: [[D1:%.+]] = moore.read [[DATAWIRE]] : <l8>
  // CHECK-NEXT: [[D1_INT:%.+]] = moore.logic_to_int [[D1]] : l8
  // CHECK-NEXT: [[DB:%.+]] = moore.to_builtin_int [[D1_INT]] : i8
  // CHECK-NEXT: [[PAST:%.+]] = ltl.past [[DB]], 1 clk [[CLK_I1]] : i8
  // CHECK-NEXT: [[PAST_INT:%.+]] = moore.from_builtin_int [[PAST]] : i8
  // CHECK-NEXT: [[WIRE_PAST_VAL:%.+]] = moore.int_to_logic [[PAST_INT]] : i8
  wire [7:0] wire_past = $past(data_i);
endmodule

// CHECK-LABEL: func.func private @BitVectorPackedBuiltins(
// CHECK-SAME: [[S:%[^ ,]+]]: !moore.struct<{a: l4, b: l4}>,
// CHECK-SAME: [[A:%[^ ,]+]]: !moore.array<2 x l4>)
function void BitVectorPackedBuiltins(
    struct packed { logic [3:0] a; logic [3:0] b; } s,
    logic [1:0][3:0] arr);
  bit result;
  int cnt;

  // CHECK: [[SBV:%.+]] = moore.packed_to_sbv [[S]] : struct<{a: l4, b: l4}>
  // CHECK-NEXT: [[RED:%.+]] = moore.reduce_xor [[SBV]] : l8 -> l1
  // CHECK-NEXT: [[X:%.+]] = moore.constant bX : l1
  // CHECK-NEXT: moore.case_eq [[RED]], [[X]] : l1
  result = $isunknown(s);

  // CHECK: [[SBV2:%.+]] = moore.packed_to_sbv [[S]] : struct<{a: l4, b: l4}>
  // CHECK: moore.reduce_xor [[SBV2]] : l8 -> l1
  // CHECK: comb.icmp eq
  result = $onehot0(s);

  // CHECK: [[SBV3:%.+]] = moore.packed_to_sbv [[S]] : struct<{a: l4, b: l4}>
  // CHECK: moore.reduce_xor [[SBV3]] : l8 -> l1
  // CHECK: comb.icmp ne
  result = $onehot(s);

  // CHECK: [[SBV4:%.+]] = moore.packed_to_sbv [[S]] : struct<{a: l4, b: l4}>
  // CHECK-NEXT: moore.logic_to_int [[SBV4]] : l8
  cnt = $countones(s);

  // CHECK: [[SBV5:%.+]] = moore.packed_to_sbv [[A]] : array<2 x l4>
  // CHECK-NEXT: [[RED2:%.+]] = moore.reduce_xor [[SBV5]] : l8 -> l1
  // CHECK-NEXT: [[X2:%.+]] = moore.constant bX : l1
  // CHECK-NEXT: moore.case_eq [[RED2]], [[X2]] : l1
  result = $isunknown(arr);
endfunction

// CHECK-LABEL: func.func private @StringBuiltins(
// CHECK-SAME: [[STR:%.+]]: !moore.string,
// CHECK-SAME: [[INT:%.+]]: !moore.i32,
// CHECK-SAME: [[Other:%.+]]: !moore.string) {
function void StringBuiltins(string string_in, int int_in, string other);
  // CHECK: [[VAR:%.+]] = moore.variable [[STR]] : <string>
  // CHECK: [[READ:%.+]] = moore.read [[VAR]] : <string>
  // CHECK: [[LEN:%.+]] = moore.string.len [[READ]]
  dummyA(string_in.len());
  // CHECK: moore.string.put [[VAR]]{{\[}}[[INT]]{{\]}}, {{%.+}} : <string>
  string_in.putc(int_in, "A");
  // CHECK: [[GET:%.+]] = moore.string.get {{%.+}}{{\[}}{{%.+}}{{\]}}
  dummyE(string_in.getc(int_in));
  // CHECK: [[UPPER:%.+]] = moore.string.toupper {{%.+}}
  dummyD(string_in.toupper());
  // CHECK: [[LOWER:%.+]] = moore.string.tolower {{%.+}}
  dummyD(string_in.tolower());
  // CHECK: [[CMP:%.+]] = moore.string.compare {{%.+}}, [[Other]]
  dummyA(string_in.compare(other));
  // CHECK: [[ICMP:%.+]] = moore.string.icompare {{%.+}}, [[Other]]
  dummyA(string_in.icompare(other));
  // CHECK: [[SUBSTR:%.+]] = moore.string.substr {{%.+}}{{\[}}{{%.+}} : {{%.+}}{{\]}}
  dummyD(string_in.substr(0, 2));
  // CHECK: [[ATOI:%.+]] = moore.string.atoi {{%.+}} : l32
  // CHECK: moore.logic_to_int [[ATOI]]
  dummyA(string_in.atoi());
  // CHECK: [[ATOHEX:%.+]] = moore.string.atohex {{%.+}} : l32
  // CHECK: moore.logic_to_int [[ATOHEX]]
  dummyA(string_in.atohex());
  // CHECK: [[ATOOCT:%.+]] = moore.string.atooct {{%.+}} : l32
  // CHECK: moore.logic_to_int [[ATOOCT]]
  dummyA(string_in.atooct());
  // CHECK: [[ATOBIN:%.+]] = moore.string.atobin {{%.+}} : l32
  // CHECK: moore.logic_to_int [[ATOBIN]]
  dummyA(string_in.atobin());
  // CHECK: [[ATOREAL:%.+]] = moore.string.atoreal {{%.+}} : f64
  dummyB(string_in.atoreal());
  // CHECK: moore.string.itoa [[VAR]], {{%.+}} : <string>, l32
  string_in.itoa(int_in);
  // CHECK: moore.string.hextoa [[VAR]], {{%.+}} : <string>, l32
  string_in.hextoa(int_in);
  // CHECK: moore.string.octtoa [[VAR]], {{%.+}} : <string>, l32
  string_in.octtoa(int_in);
  // CHECK: moore.string.bintoa [[VAR]], {{%.+}} : <string>, l32
  string_in.bintoa(int_in);
  // CHECK: moore.string.realtoa [[VAR]], {{%.+}} : <string>, f64
  string_in.realtoa(1.5);
endfunction

// IEEE 1800-2017 § 21.3 "File I/O system tasks and functions"
// CHECK-LABEL: func.func private @FileIOBuiltins(
// CHECK-SAME: [[FD_INT:%[^ ,]+]]: !moore.i32
// CHECK-SAME: [[FD_INTEGER:%[^ ,]+]]: !moore.l32
function void FileIOBuiltins(int fd_int, integer fd_integer);
  int fd;

  // CHECK: [[FNAME:%.+]] = moore.constant_string "file.txt" : i64
  // CHECK-NEXT: [[FNAME_S:%.+]] = moore.int_to_string [[FNAME]] : i64
  // CHECK-NEXT: [[FD1:%.+]] = moore.builtin.fopen [[FNAME_S]]
  fd = $fopen("file.txt");

  // CHECK: [[F:%.+]] = moore.constant_string "f" : i8
  // CHECK-NEXT: [[FS:%.+]] = moore.int_to_string [[F]] : i8
  // CHECK-NEXT: moore.builtin.fopen [[FS]] mode = r
  fd = $fopen("f", "r");

  // CHECK: [[F:%.+]] = moore.constant_string "f" : i8
  // CHECK-NEXT: [[FS:%.+]] = moore.int_to_string [[F]] : i8
  // CHECK-NEXT: moore.builtin.fopen [[FS]] mode = r
  fd = $fopen("f", "rb");

  // CHECK: [[F:%.+]] = moore.constant_string "f" : i8
  // CHECK-NEXT: [[FS:%.+]] = moore.int_to_string [[F]] : i8
  // CHECK-NEXT: moore.builtin.fopen [[FS]] mode = w
  fd = $fopen("f", "w");

  // CHECK: [[F:%.+]] = moore.constant_string "f" : i8
  // CHECK-NEXT: [[FS:%.+]] = moore.int_to_string [[F]] : i8
  // CHECK-NEXT: moore.builtin.fopen [[FS]] mode = w
  fd = $fopen("f", "wb");

  // CHECK: [[F:%.+]] = moore.constant_string "f" : i8
  // CHECK-NEXT: [[FS:%.+]] = moore.int_to_string [[F]] : i8
  // CHECK-NEXT: moore.builtin.fopen [[FS]] mode = a
  fd = $fopen("f", "a");

  // CHECK: [[F:%.+]] = moore.constant_string "f" : i8
  // CHECK-NEXT: [[FS:%.+]] = moore.int_to_string [[F]] : i8
  // CHECK-NEXT: moore.builtin.fopen [[FS]] mode = a
  fd = $fopen("f", "ab");

  // CHECK: [[F:%.+]] = moore.constant_string "f" : i8
  // CHECK-NEXT: [[FS:%.+]] = moore.int_to_string [[F]] : i8
  // CHECK-NEXT: moore.builtin.fopen [[FS]] mode = r_update
  fd = $fopen("f", "r+");

  // CHECK: [[F:%.+]] = moore.constant_string "f" : i8
  // CHECK-NEXT: [[FS:%.+]] = moore.int_to_string [[F]] : i8
  // CHECK-NEXT: moore.builtin.fopen [[FS]] mode = r_update
  fd = $fopen("f", "r+b");

  // CHECK: [[F:%.+]] = moore.constant_string "f" : i8
  // CHECK-NEXT: [[FS:%.+]] = moore.int_to_string [[F]] : i8
  // CHECK-NEXT: moore.builtin.fopen [[FS]] mode = r_update
  fd = $fopen("f", "rb+");

  // CHECK: [[F:%.+]] = moore.constant_string "f" : i8
  // CHECK-NEXT: [[FS:%.+]] = moore.int_to_string [[F]] : i8
  // CHECK-NEXT: moore.builtin.fopen [[FS]] mode = w_update
  fd = $fopen("f", "w+");

  // CHECK: [[F:%.+]] = moore.constant_string "f" : i8
  // CHECK-NEXT: [[FS:%.+]] = moore.int_to_string [[F]] : i8
  // CHECK-NEXT: moore.builtin.fopen [[FS]] mode = w_update
  fd = $fopen("f", "w+b");

  // CHECK: [[F:%.+]] = moore.constant_string "f" : i8
  // CHECK-NEXT: [[FS:%.+]] = moore.int_to_string [[F]] : i8
  // CHECK-NEXT: moore.builtin.fopen [[FS]] mode = w_update
  fd = $fopen("f", "wb+");

  // CHECK: [[F:%.+]] = moore.constant_string "f" : i8
  // CHECK-NEXT: [[FS:%.+]] = moore.int_to_string [[F]] : i8
  // CHECK-NEXT: moore.builtin.fopen [[FS]] mode = a_update
  fd = $fopen("f", "a+");

  // CHECK: [[F:%.+]] = moore.constant_string "f" : i8
  // CHECK-NEXT: [[FS:%.+]] = moore.int_to_string [[F]] : i8
  // CHECK-NEXT: moore.builtin.fopen [[FS]] mode = a_update
  fd = $fopen("f", "a+b");

  // CHECK: [[F:%.+]] = moore.constant_string "f" : i8
  // CHECK-NEXT: [[FS:%.+]] = moore.int_to_string [[F]] : i8
  // CHECK-NEXT: moore.builtin.fopen [[FS]] mode = a_update
  fd = $fopen("f", "ab+");

  // CHECK: [[FDVAL:%.+]] = moore.read {{%.+}} : <i32>
  // CHECK-NEXT: moore.builtin.fclose [[FDVAL]]
  $fclose(fd);

  // CHECK: [[FDVAL:%.+]] = moore.read {{%.+}} : <i32>
  // CHECK-NEXT: moore.builtin.fflush [[FDVAL]]
  $fflush(fd);

  // CHECK: moore.builtin.fflush
  $fflush();
endfunction

// IEEE 1800-2017 § 21.3.2 "File output system tasks"
// CHECK-LABEL: func.func private @FileDisplayBuiltins(
// CHECK-SAME: [[FD:%[^ ,]+]]: !moore.i32
// CHECK-SAME: [[X:%[^ ,]+]]: !moore.i32
function void FileDisplayBuiltins(int fd, int x);
  // $fwrite with no message
  // CHECK-NOT: moore.builtin.fdisplay
  $fwrite(fd);

  // $fdisplay with no args
  // CHECK: [[TMP:%.+]] = moore.fmt.literal "\0A"
  // CHECK: moore.builtin.fdisplay [[FD]], [[TMP]]
  $fdisplay(fd);

  // CHECK: [[TMP:%.+]] = moore.fmt.literal "hello"
  // CHECK: moore.builtin.fdisplay [[FD]], [[TMP]]
  $fwrite(fd, "hello");

  // $fdisplay adds \0A newline
  // CHECK: [[TMP1:%.+]] = moore.fmt.literal "hello"
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.fdisplay [[FD]], [[TMP3]]
  $fdisplay(fd, "hello");

  // CHECK: [[TMP:%.+]] = moore.fmt.int decimal [[X]], align right, pad space signed : i32
  // CHECK: moore.builtin.fdisplay [[FD]], [[TMP]]
  $fwrite(fd, x);
  // CHECK: [[TMP:%.+]] = moore.fmt.int binary [[X]], align right, pad zero : i32
  // CHECK: moore.builtin.fdisplay [[FD]], [[TMP]]
  $fwriteb(fd, x);
  // CHECK: [[TMP:%.+]] = moore.fmt.int octal [[X]], align right, pad zero : i32
  // CHECK: moore.builtin.fdisplay [[FD]], [[TMP]]
  $fwriteo(fd, x);
  // CHECK: [[TMP:%.+]] = moore.fmt.int hex_lower [[X]], align right, pad zero : i32
  // CHECK: moore.builtin.fdisplay [[FD]], [[TMP]]
  $fwriteh(fd, x);

  // CHECK: [[TMP1:%.+]] = moore.fmt.int decimal [[X]], align right, pad space signed : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.fdisplay [[FD]], [[TMP3]]
  $fdisplay(fd, x);
  // CHECK: [[TMP1:%.+]] = moore.fmt.int binary [[X]], align right, pad zero : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.fdisplay [[FD]], [[TMP3]]
  $fdisplayb(fd, x);
  // CHECK: [[TMP1:%.+]] = moore.fmt.int octal [[X]], align right, pad zero : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.fdisplay [[FD]], [[TMP3]]
  $fdisplayo(fd, x);
  // CHECK: [[TMP1:%.+]] = moore.fmt.int hex_lower [[X]], align right, pad zero : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.fdisplay [[FD]], [[TMP3]]
  $fdisplayh(fd, x);

endfunction

// IEEE 1800-2017 § 21.6 "Command line input"
// CHECK-LABEL: func.func private @PlusArgsBuiltins(
function void PlusArgsBuiltins();
  bit rv;
  int val;

  // CHECK: [[T:%.+]] = moore.builtin.plusargs_test "FOO" : i1
  rv = $test$plusargs("FOO");

  // CHECK: [[FOUND:%.+]], [[RESULT:%.+]] = moore.builtin.plusargs_value "BAR=%d" : i1, i32
  // CHECK: moore.blocking_assign {{%.+}}, [[RESULT]] : i32
  rv = $value$plusargs("BAR=%d", val);
endfunction

// CHECK-LABEL: func.func private @FScanfIntegerSpecifiers(
// CHECK-SAME:  [[FD:%[^ ,]+]]: !moore.i32
function void FScanfIntegerSpecifiers(int fd);
  int x;
  int res;

  // CHECK: [[XVAR:%.+]] = moore.variable : <i32>
  // CHECK: moore.variable : <i32>

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.int [[C]] decimal : i32, !moore.i1
  // CHECK: [[MV:%.+]] = moore.from_builtin_int [[V]] : i32
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[XVAR]], [[MV]] : i32
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%d", x);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.int [[C]] decimal : i32, !moore.i1
  // CHECK: [[MV:%.+]] = moore.from_builtin_int [[V]] : i32
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[XVAR]], [[MV]] : i32
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%D", x);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.int [[C]] binary : i32, !moore.i1
  // CHECK: [[MV:%.+]] = moore.from_builtin_int [[V]] : i32
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[XVAR]], [[MV]] : i32
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%b", x);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.int [[C]] binary : i32, !moore.i1
  // CHECK: [[MV:%.+]] = moore.from_builtin_int [[V]] : i32
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[XVAR]], [[MV]] : i32
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%B", x);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.int [[C]] octal : i32, !moore.i1
  // CHECK: [[MV:%.+]] = moore.from_builtin_int [[V]] : i32
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[XVAR]], [[MV]] : i32
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%o", x);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.int [[C]] octal : i32, !moore.i1
  // CHECK: [[MV:%.+]] = moore.from_builtin_int [[V]] : i32
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[XVAR]], [[MV]] : i32
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%O", x);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.int [[C]] hex_lower : i32, !moore.i1
  // CHECK: [[MV:%.+]] = moore.from_builtin_int [[V]] : i32
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[XVAR]], [[MV]] : i32
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%h", x);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.int [[C]] hex_lower : i32, !moore.i1
  // CHECK: [[MV:%.+]] = moore.from_builtin_int [[V]] : i32
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[XVAR]], [[MV]] : i32
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%x", x);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.int [[C]] hex_upper : i32, !moore.i1
  // CHECK: [[MV:%.+]] = moore.from_builtin_int [[V]] : i32
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[XVAR]], [[MV]] : i32
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%H", x);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.int [[C]] hex_upper : i32, !moore.i1
  // CHECK: [[MV:%.+]] = moore.from_builtin_int [[V]] : i32
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[XVAR]], [[MV]] : i32
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%X", x);
endfunction

// CHECK-LABEL: func.func private @FScanfRealSpecifiers(
// CHECK-SAME:  [[FD:%[^ ,]+]]: !moore.i32
function void FScanfRealSpecifiers(int fd);
  real r;
  time t;
  int res;
  // CHECK: [[RVAR:%.+]] = moore.variable : <f64>
  // CHECK: [[TVAR:%.+]] = moore.variable : <time>
  // CHECK: moore.variable : <i32>

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.real [[C]] : !moore.f64, !moore.i1
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[RVAR]], [[V]] : f64
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%f", r);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.real [[C]] : !moore.f64, !moore.i1
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[RVAR]], [[V]] : f64
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%e", r);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.real [[C]] : !moore.f64, !moore.i1
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[RVAR]], [[V]] : f64
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%g", r);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.real [[C]] : !moore.f64, !moore.i1
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[RVAR]], [[V]] : f64
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%F", r);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.real [[C]] : !moore.f64, !moore.i1
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[RVAR]], [[V]] : f64
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%E", r);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.real [[C]] : !moore.f64, !moore.i1
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[RVAR]], [[V]] : f64
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%G", r);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.time [[C]] : !moore.time, !moore.i1
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[TVAR]], [[V]] : time
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%t", t);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.time [[C]] : !moore.time, !moore.i1
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[TVAR]], [[V]] : time
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%T", t);
endfunction

// CHECK-LABEL: func.func private @FScanfOtherSpecifiers(
// CHECK-SAME:  [[FD:%[^ ,]+]]: !moore.i32
function void FScanfOtherSpecifiers(int fd);
  string s;
  byte c;
  int raw;
  int res;

  // CHECK: [[SVAR:%.+]] = moore.variable : <string>
  // CHECK: [[CVAR:%.+]] = moore.variable : <i8>
  // CHECK: [[RVAR:%.+]] = moore.variable : <i32>
  // CHECK: moore.variable : <i32>

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.str [[C]] : !moore.string, !moore.i1
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[SVAR]], [[V]] : string
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%s", s);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.str [[C]] : !moore.string, !moore.i1
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[SVAR]], [[V]] : string
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%S", s);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.char [[C]] : i8, !moore.i1
  // CHECK: [[MV:%.+]] = moore.from_builtin_int [[V]] : i8
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[CVAR]], [[MV]] : i8
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%c", c);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.unformatted [[C]] : i32, !moore.i1
  // CHECK-NOT: four_value
  // CHECK: [[MV:%.+]] = moore.from_builtin_int [[V]] : i32
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[RVAR]], [[MV]] : i32
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%u", raw);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.unformatted [[C]] four_value : i32, !moore.i1
  // CHECK: [[MV:%.+]] = moore.from_builtin_int [[V]] : i32
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[RVAR]], [[MV]] : i32
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%z", raw);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.unformatted [[C]] four_value : i32, !moore.i1
  // CHECK: [[MV:%.+]] = moore.from_builtin_int [[V]] : i32
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[RVAR]], [[MV]] : i32
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%Z", raw);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]] = moore.scan.hier_path_match [[C]]
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%m");

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]] = moore.scan.hier_path_match [[C]]
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%M");

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]] = moore.scan.literal [[C]] "%"
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%%");
endfunction

// CHECK-LABEL: func.func private @FScanfModifiers(
// CHECK-SAME:  [[FD:%[^ ,]+]]: !moore.i32
function void FScanfModifiers(int fd);
  int x;
  string s;
  real r;
  int res;

  // CHECK: [[IVAR:%.+]] = moore.variable : <i32>
  // CHECK: [[SVAR:%.+]] = moore.variable : <string>
  // CHECK: [[RVAR:%.+]] = moore.variable : <f64>
  // CHECK: moore.variable : <i32>

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]] = moore.scan.int [[C]] decimal
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%*d");

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]] = moore.scan.int [[C]] binary
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%*b");

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]] = moore.scan.str [[C]]
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%*s");

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]] = moore.scan.real [[C]]
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%*f");

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]] = moore.scan.char [[C]]
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%*c");

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.int [[C]] decimal width 8 : i32, !moore.i1
  // CHECK: [[MV:%.+]] = moore.from_builtin_int [[V]] : i32
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[IVAR]], [[MV]] : i32
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%8d", x);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.str [[C]] width 16 : !moore.string, !moore.i1
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[SVAR]], [[V]] : string
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%16s", s);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.real [[C]] width 10 : !moore.f64, !moore.i1
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[RVAR]], [[V]] : f64
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%10f", r);

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]] = moore.scan.int [[C]] decimal width 42
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%*42d");

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]] = moore.scan.str [[C]] width 42
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%*42s");

  // CHECK: [[C:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[R:%.+]] = moore.scan.int [[C]] hex_lower width 42
  // CHECK: moore.scan.end [[R]]
  res = $fscanf(fd, "%*42h");
endfunction

// CHECK-LABEL: func.func private @FScanfComposition(
// CHECK-SAME:  [[FD:%[^ ,]+]]: !moore.i32
function void FScanfComposition(int fd);
  int a;
  string b;
  real r;
  int res;

  // CHECK: [[AVAR:%.+]] = moore.variable : <i32>
  // CHECK: [[BVAR:%.+]] = moore.variable : <string>
  // CHECK: [[RVAR:%.+]] = moore.variable : <f64>
  // CHECK: moore.variable : <i32>

  // CHECK: [[C0:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[C1:%.+]], [[V1:%.+]], [[M1:%.+]] = moore.scan.int [[C0]] decimal : i32, !moore.i1
  // CHECK: [[MV1:%.+]] = moore.from_builtin_int [[V1]] : i32
  // CHECK: [[C2:%.+]], [[V2:%.+]], [[M2:%.+]] = moore.scan.str [[C1]] : !moore.string, !moore.i1
  // CHECK: [[CB1:%.+]] = moore.to_builtin_int [[M1]] : i1
  // CHECK: cf.cond_br [[CB1]], ^[[A1:.+]], ^[[K1:.+]]
  // CHECK: ^[[A1]]:
  // CHECK: moore.blocking_assign [[AVAR]], [[MV1]] : i32
  // CHECK: ^[[K1]]:
  // CHECK: [[CB2:%.+]] = moore.to_builtin_int [[M2]] : i1
  // CHECK: cf.cond_br [[CB2]], ^[[A2:.+]], ^[[K2:.+]]
  // CHECK: ^[[A2]]:
  // CHECK: moore.blocking_assign [[BVAR]], [[V2]] : string
  // CHECK: ^[[K2]]:
  // CHECK: moore.scan.end [[C2]]
  res = $fscanf(fd, "%d%s", a, b);

  // CHECK: [[C0:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[C1:%.+]], [[V1:%.+]], [[M1:%.+]] = moore.scan.int [[C0]] decimal : i32, !moore.i1
  // CHECK: [[MV1:%.+]] = moore.from_builtin_int [[V1]] : i32
  // CHECK: [[C2:%.+]], [[V2:%.+]], [[M2:%.+]] = moore.scan.real [[C1]] : !moore.f64, !moore.i1
  // CHECK: [[C3:%.+]], [[V3:%.+]], [[M3:%.+]] = moore.scan.str [[C2]] : !moore.string, !moore.i1
  // CHECK: [[CB1:%.+]] = moore.to_builtin_int [[M1]] : i1
  // CHECK: cf.cond_br [[CB1]], ^[[A1:.+]], ^[[K1:.+]]
  // CHECK: ^[[A1]]:
  // CHECK: moore.blocking_assign [[AVAR]], [[MV1]] : i32
  // CHECK: ^[[K1]]:
  // CHECK: [[CB2:%.+]] = moore.to_builtin_int [[M2]] : i1
  // CHECK: cf.cond_br [[CB2]], ^[[A2:.+]], ^[[K2:.+]]
  // CHECK: ^[[A2]]:
  // CHECK: moore.blocking_assign [[RVAR]], [[V2]] : f64
  // CHECK: ^[[K2]]:
  // CHECK: [[CB3:%.+]] = moore.to_builtin_int [[M3]] : i1
  // CHECK: cf.cond_br [[CB3]], ^[[A3:.+]], ^[[K3:.+]]
  // CHECK: ^[[A3]]:
  // CHECK: moore.blocking_assign [[BVAR]], [[V3]] : string
  // CHECK: ^[[K3]]:
  // CHECK: moore.scan.end [[C3]]
  res = $fscanf(fd, "%d%f%s", a, r, b);

  // CHECK: [[C0:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[C1:%.+]] = moore.scan.literal [[C0]] "val="
  // CHECK: [[C2:%.+]], [[V1:%.+]], [[M1:%.+]] = moore.scan.int [[C1]] decimal : i32, !moore.i1
  // CHECK: [[MV1:%.+]] = moore.from_builtin_int [[V1]] : i32
  // CHECK: [[CB1:%.+]] = moore.to_builtin_int [[M1]] : i1
  // CHECK: cf.cond_br [[CB1]], ^[[A1:.+]], ^[[K1:.+]]
  // CHECK: ^[[A1]]:
  // CHECK: moore.blocking_assign [[AVAR]], [[MV1]] : i32
  // CHECK: ^[[K1]]:
  // CHECK: moore.scan.end [[C2]]
  res = $fscanf(fd, "val=%d", a);

  // CHECK: [[C0:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[C1:%.+]], [[V1:%.+]], [[M1:%.+]] = moore.scan.int [[C0]] decimal : i32, !moore.i1
  // CHECK: [[MV1:%.+]] = moore.from_builtin_int [[V1]] : i32
  // CHECK: [[C2:%.+]] = moore.scan.literal [[C1]] " end"
  // CHECK: [[CB1:%.+]] = moore.to_builtin_int [[M1]] : i1
  // CHECK: cf.cond_br [[CB1]], ^[[A1:.+]], ^[[K1:.+]]
  // CHECK: ^[[A1]]:
  // CHECK: moore.blocking_assign [[AVAR]], [[MV1]] : i32
  // CHECK: ^[[K1]]:
  // CHECK: moore.scan.end [[C2]]
  res = $fscanf(fd, "%d end", a);

  // CHECK: [[C0:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[C1:%.+]], [[V1:%.+]], [[M1:%.+]] = moore.scan.int [[C0]] decimal : i32, !moore.i1
  // CHECK: [[MV1:%.+]] = moore.from_builtin_int [[V1]] : i32
  // CHECK: [[C2:%.+]] = moore.scan.literal [[C1]] " , "
  // CHECK: [[C3:%.+]], [[V2:%.+]], [[M2:%.+]] = moore.scan.str [[C2]] : !moore.string, !moore.i1
  // CHECK: [[CB1:%.+]] = moore.to_builtin_int [[M1]] : i1
  // CHECK: cf.cond_br [[CB1]], ^[[A1:.+]], ^[[K1:.+]]
  // CHECK: ^[[A1]]:
  // CHECK: moore.blocking_assign [[AVAR]], [[MV1]] : i32
  // CHECK: ^[[K1]]:
  // CHECK: [[CB2:%.+]] = moore.to_builtin_int [[M2]] : i1
  // CHECK: cf.cond_br [[CB2]], ^[[A2:.+]], ^[[K2:.+]]
  // CHECK: ^[[A2]]:
  // CHECK: moore.blocking_assign [[BVAR]], [[V2]] : string
  // CHECK: ^[[K2]]:
  // CHECK: moore.scan.end [[C3]]
  res = $fscanf(fd, "%d , %s", a, b);

  // CHECK: [[C0:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[C1:%.+]] = moore.scan.int [[C0]] decimal
  // CHECK: [[C2:%.+]], [[V1:%.+]], [[M1:%.+]] = moore.scan.str [[C1]] : !moore.string, !moore.i1
  // CHECK: [[CB1:%.+]] = moore.to_builtin_int [[M1]] : i1
  // CHECK: cf.cond_br [[CB1]], ^[[A1:.+]], ^[[K1:.+]]
  // CHECK: ^[[A1]]:
  // CHECK: moore.blocking_assign [[BVAR]], [[V1]] : string
  // CHECK: ^[[K1]]:
  // CHECK: moore.scan.end [[C2]]
  res = $fscanf(fd, "%*d%s", b);

  // CHECK: [[C0:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[C1:%.+]] = moore.scan.literal [[C0]] "prefix"
  // CHECK: moore.scan.end [[C1]]
  res = $fscanf(fd, "prefix");

  // CHECK: [[C0:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: [[C1:%.+]] = moore.scan.literal [[C0]] " "
  // CHECK: moore.scan.end [[C1]]
  res = $fscanf(fd, " ");

  // CHECK: [[C0:%.+]] = moore.scan.begin_fscanf [[FD]]
  // CHECK: moore.scan.end [[C0]]
  res = $fscanf(fd, "");
endfunction

// CHECK-LABEL: func.func private @SScanfBuiltins(
// CHECK-SAME:  [[SRC:%[^ ,]+]]: !moore.string
function void SScanfBuiltins(string src);
  int x;
  string s;
  real r;
  int res;

  // CHECK: [[XVAR:%.+]] = moore.variable : <i32>
  // CHECK: [[SVAR:%.+]] = moore.variable : <string>
  // CHECK: [[RVAR:%.+]] = moore.variable : <f64>
  // CHECK: moore.variable : <i32>

  // CHECK: [[C:%.+]] = moore.scan.begin_sscanf [[SRC]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.int [[C]] decimal : i32, !moore.i1
  // CHECK: [[MV:%.+]] = moore.from_builtin_int [[V]] : i32
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[XVAR]], [[MV]] : i32
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $sscanf(src, "%d", x);

  // CHECK: [[C:%.+]] = moore.scan.begin_sscanf [[SRC]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.str [[C]] : !moore.string, !moore.i1
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[SVAR]], [[V]] : string
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $sscanf(src, "%s", s);

  // CHECK: [[C0:%.+]] = moore.scan.begin_sscanf [[SRC]]
  // CHECK: [[C1:%.+]], [[V1:%.+]], [[M1:%.+]] = moore.scan.int [[C0]] decimal : i32, !moore.i1
  // CHECK: [[MV1:%.+]] = moore.from_builtin_int [[V1]] : i32
  // CHECK: [[C2:%.+]] = moore.scan.literal [[C1]] " "
  // CHECK: [[C3:%.+]], [[V2:%.+]], [[M2:%.+]] = moore.scan.real [[C2]] : !moore.f64, !moore.i1
  // CHECK: [[CB1:%.+]] = moore.to_builtin_int [[M1]] : i1
  // CHECK: cf.cond_br [[CB1]], ^[[A1:.+]], ^[[K1:.+]]
  // CHECK: ^[[A1]]:
  // CHECK: moore.blocking_assign [[XVAR]], [[MV1]] : i32
  // CHECK: ^[[K1]]:
  // CHECK: [[CB2:%.+]] = moore.to_builtin_int [[M2]] : i1
  // CHECK: cf.cond_br [[CB2]], ^[[A2:.+]], ^[[K2:.+]]
  // CHECK: ^[[A2]]:
  // CHECK: moore.blocking_assign [[RVAR]], [[V2]] : f64
  // CHECK: ^[[K2]]:
  // CHECK: moore.scan.end [[C3]]
  res = $sscanf(src, "%d %f", x, r);

  // CHECK: [[C:%.+]] = moore.scan.begin_sscanf [[SRC]]
  // CHECK: [[R:%.+]] = moore.scan.int [[C]] decimal width 8
  // CHECK: moore.scan.end [[R]]
  res = $sscanf(src, "%*8d");

  // CHECK: [[C:%.+]] = moore.scan.begin_sscanf [[SRC]]
  // CHECK: [[R:%.+]], [[V:%.+]], [[M:%.+]] = moore.scan.unformatted [[C]] four_value : i32, !moore.i1
  // CHECK: [[MV:%.+]] = moore.from_builtin_int [[V]] : i32
  // CHECK: [[CB:%.+]] = moore.to_builtin_int [[M]] : i1
  // CHECK: cf.cond_br [[CB]], ^[[ASSIGN:.+]], ^[[CONT:.+]]
  // CHECK: ^[[ASSIGN]]:
  // CHECK: moore.blocking_assign [[XVAR]], [[MV]] : i32
  // CHECK: ^[[CONT]]:
  // CHECK: moore.scan.end [[R]]
  res = $sscanf(src, "%z", x);

  // CHECK: [[C0:%.+]] = moore.scan.begin_sscanf [[SRC]]
  // CHECK: moore.scan.end [[C0]]
  res = $sscanf(src, "");
endfunction

// IEEE 1800-2017 § 21.4 "Loading memory array data from a file"
// CHECK-LABEL: moore.module @ReadMemBuiltins
module ReadMemBuiltins;
  logic [7:0] mem [1:256];
  logic [7:0] memDesc [255:0];
  logic [31:0] mem3 [0:2][0:4][5:8];
  logic [7:0] memS [0:255];
  logic [7:0] qmem [$];
  typedef enum logic [1:0] {A=0, B=1, C=2} e_t;
  e_t emem [0:7];
  typedef struct packed { logic [3:0] a; logic [3:0] b; } s_t;
  s_t smem [0:7];
  int startDyn, finishDyn;
  initial begin
    // CHECK: moore.builtin.readmem hex %{{[0-9]+}}, %mem {dimDescending = array<i1: false>, dimLows = array<i64: 1>} : !moore.ref<uarray<256 x l8>>
    $readmemh("mem.data", mem);
    // CHECK: moore.builtin.readmem bin %{{[0-9]+}}, %mem start = %{{[0-9]+}} {dimDescending = array<i1: false>, dimLows = array<i64: 1>} : !moore.ref<uarray<256 x l8>>
    $readmemb("mem.data", mem, 16);
    // CHECK: moore.builtin.readmem hex %{{[0-9]+}}, %mem start = %{{[0-9]+}} finish = %{{[0-9]+}} {dimDescending = array<i1: false>, dimLows = array<i64: 1>} : !moore.ref<uarray<256 x l8>>
    $readmemh("mem.data", mem, 128, 1);
    // CHECK: moore.builtin.readmem hex %{{[0-9]+}}, %memDesc {dimDescending = array<i1: true>, dimLows = array<i64: 0>} : !moore.ref<uarray<256 x l8>>
    $readmemh("mem.data", memDesc);
    // CHECK: moore.builtin.readmem hex %{{[0-9]+}}, %mem3 {dimDescending = array<i1: false, false, false>, dimLows = array<i64: 0, 0, 5>} : !moore.ref<uarray<3 x uarray<5 x uarray<4 x l32>>>>
    $readmemh("mem.data", mem3);
    // CHECK: [[SUBARR:%[0-9]+]] = moore.extract_ref %mem3 from 1
    // CHECK: moore.builtin.readmem hex %{{[0-9]+}}, [[SUBARR]] {dimDescending = array<i1: false, false>, dimLows = array<i64: 0, 5>} : !moore.ref<uarray<5 x uarray<4 x l32>>>
    $readmemh("mem.data", mem3[1]);
    // CHECK: moore.builtin.readmem hex %{{[0-9]+}}, %memS slice[%{{[0-9]+}}, %{{[0-9]+}}] {dimDescending = array<i1: false>, dimLows = array<i64: 0>} : !moore.ref<uarray<256 x l8>>
    $readmemh("mem.data", memS[16:31]);
    // CHECK: moore.builtin.readmem hex %{{[0-9]+}}, %memS start = %{{[0-9]+}} finish = %{{[0-9]+}} slice[%{{[0-9]+}}, %{{[0-9]+}}] {dimDescending = array<i1: false>, dimLows = array<i64: 0>} : !moore.ref<uarray<256 x l8>>
    $readmemh("mem.data", memS[16:31], 20, 24);
    // CHECK: moore.builtin.readmem hex %{{[0-9]+}}, %qmem {dimDescending = array<i1: false>, dimLows = array<i64: 0>} : !moore.ref<queue<l8, 0>>
    $readmemh("mem.data", qmem);
    // CHECK: moore.builtin.readmem hex %{{[0-9]+}}, %emem {dimDescending = array<i1: false>, dimLows = array<i64: 0>, enumValues = array<i64: 0, 1, 2>} : !moore.ref<uarray<8 x l2>>
    $readmemh("mem.data", emem);
    // CHECK: moore.builtin.readmem hex %{{[0-9]+}}, %smem {dimDescending = array<i1: false>, dimLows = array<i64: 0>} : !moore.ref<uarray<8 x struct<{a: l4, b: l4}>>>
    $readmemh("mem.data", smem);
    // CHECK: moore.builtin.readmem hex %{{[0-9]+}}, %mem start = %{{[0-9]+}} finish = %{{[0-9]+}} {dimDescending = array<i1: false>, dimLows = array<i64: 1>} : !moore.ref<uarray<256 x l8>>
    $readmemh("mem.data", mem, startDyn, finishDyn);
  end
endmodule
