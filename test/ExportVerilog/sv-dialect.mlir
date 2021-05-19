// RUN: circt-translate %s -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

// CHECK-LABEL: module M1(
rtl.module @M1(%clock : i1, %cond : i1, %val : i8) {
  %wire42 = sv.wire : !rtl.inout<i42>

  // CHECK:      always @(posedge clock) begin
  // CHECK-NEXT:   `ifndef SYNTHESIS
  sv.always posedge %clock {
    sv.ifdef.procedural "SYNTHESIS" {
    } else {
  // CHECK-NEXT:     if (PRINTF_COND_ & 1'bx & 1'bz & 1'bz & cond)
      %tmp = sv.verbatim.expr "PRINTF_COND_" : () -> i1
      %tmp1 = sv.constantX : i1
      %tmp2 = sv.constantZ : i1
      %tmp3 = comb.and %tmp, %tmp1, %tmp2, %tmp2, %cond : i1
      sv.if %tmp3 {
  // CHECK-NEXT:       $fwrite(32'h80000002, "Hi\n");
        sv.fwrite "Hi\n"
      }

      // CHECK-NEXT: if (!(clock | cond))
      // CHECK-NEXT:   $fwrite(32'h80000002, "Bye\n");
      %tmp4 = comb.or %clock, %cond : i1
      sv.if %tmp4 {
      } else {
        sv.fwrite "Bye\n"
      }
  // CHECK-NEXT:   `endif
  // CHECK-NEXT: end // always @(posedge)
    }
  }

  // CHECK-NEXT: always @(negedge clock) begin
  // CHECK-NEXT: end // always @(negedge)
  sv.always negedge %clock {
  }

  // CHECK-NEXT: always @(edge clock) begin
  // CHECK-NEXT: end // always @(edge)
  sv.always edge %clock {
  }

  // CHECK-NEXT: always @* begin
  // CHECK-NEXT: end // always
  sv.always {
  }

  // CHECK-NEXT: always @(posedge clock or negedge cond) begin
  // CHECK-NEXT: end // always @(posedge, negedge)
  sv.always posedge %clock, negedge %cond {
  }

  // CHECK-NEXT: always_ff @(posedge clock)
  // CHECK-NEXT:   $fwrite(32'h80000002, "Yo\n");
  sv.alwaysff(posedge %clock) {
    sv.fwrite "Yo\n"
  }
  
  // CHECK-NEXT: always_ff @(posedge clock) begin
  // CHECK-NEXT:   if (cond)
  // CHECK-NEXT:     $fwrite(32'h80000002, "Sync Reset Block\n")
  // CHECK-NEXT:   else
  // CHECK-NEXT:     $fwrite(32'h80000002, "Sync Main Block\n");
  // CHECK-NEXT: end // always_ff @(posedge)
  sv.alwaysff(posedge %clock) {
    sv.fwrite "Sync Main Block\n"
  } ( syncreset : posedge %cond) {
    sv.fwrite "Sync Reset Block\n"
  }

  // CHECK-NEXT: always_ff @(posedge clock or negedge cond) begin
  // CHECK-NEXT:   if (!cond)
  // CHECK-NEXT:     $fwrite(32'h80000002, "Async Reset Block\n");
  // CHECK-NEXT:   else
  // CHECK-NEXT:     $fwrite(32'h80000002, "Async Main Block\n");
  // CHECK-NEXT: end // always_ff @(posedge or negedge)
  sv.alwaysff(posedge %clock) {
    sv.fwrite "Async Main Block\n"
  } ( asyncreset : negedge %cond) {
    sv.fwrite "Async Reset Block\n"
  } 

  // CHECK-NEXT:  initial begin
  sv.initial {
    // CHECK-NEXT:   if (cond)
    sv.if %cond {
      %c42 = rtl.constant 42 : i42

      // CHECK-NEXT: wire42 = 42'h2A;
      sv.bpassign %wire42, %c42 : i42
    }

    // CHECK-NEXT:   if (cond)
    // CHECK-NOT: begin
    sv.if %cond {
      %c42 = rtl.constant 42 : i8
      %add = comb.add %val, %c42 : i8

      // CHECK-NEXT: $fwrite(32'h80000002, "Inlined! %x\n", val + 8'h2A);
      sv.fwrite "Inlined! %x\n"(%add) : i8
    }

    // begin/end required here to avoid else-confusion.  

    // CHECK-NEXT:   if (cond) begin
    sv.if %cond {
      // CHECK-NEXT: if (clock)
      sv.if %clock {
        // CHECK-NEXT: $fwrite(32'h80000002, "Inside Block\n");
        sv.fwrite "Inside Block\n"
      }
      // CHECK-NEXT: end
    } else { // CHECK-NEXT: else
      // CHECK-NOT: begin
      // CHECK-NEXT: $fwrite(32'h80000002, "Else Block\n");
      sv.fwrite "Else Block\n"
    }

    // CHECK-NEXT:   if (cond) begin
    sv.if %cond {
      // CHECK-NEXT:     $fwrite(32'h80000002, "Hi\n");
      sv.fwrite "Hi\n"

      // CHECK-NEXT:     $fwrite(32'h80000002, "Bye %x\n", val + val);
      %tmp = comb.add %val, %val : i8
      sv.fwrite "Bye %x\n"(%tmp) : i8

      // CHECK-NEXT:     assert(cond);
      sv.assert %cond : i1
      // CHECK-NEXT:     assume(cond);
      sv.assume %cond : i1
      // CHECK-NEXT:     cover(cond);
      sv.cover %cond : i1

      // CHECK-NEXT:   $fatal
      sv.fatal
      // CHECK-NEXT:   $finish
      sv.finish

      // CHECK-NEXT: Emit some stuff in verilog
      // CHECK-NEXT: Great power and responsibility!
      sv.verbatim "Emit some stuff in verilog\nGreat power and responsibility!"

      %c42 = rtl.constant 42 : i8
      %add = comb.add %val, %c42 : i8
      %c42_2 = rtl.constant 42 : i8
      %xor = comb.xor %val, %c42_2 : i8
      %text = sv.verbatim.expr "MACRO({{0}}, {{1}})" (%add, %xor): (i8,i8) -> i8

      // CHECK-NEXT: $fwrite(32'h80000002, "M: %x\n", MACRO(val + 8'h2A, val ^ 8'h2A));
      sv.fwrite "M: %x\n"(%text) : i8

    }// CHECK-NEXT:   {{end$}}
  }
  // CHECK-NEXT:  end // initial


  // CHECK-NEXT: initial
  // CHECK-NOT: begin
  sv.initial {
    // CHECK-NEXT: $fatal
    sv.fatal
  }

  // CHECK-NEXT: initial begin
  sv.initial {
    %thing = sv.verbatim.expr "THING" : () -> i42
    // CHECK-NEXT: wire42 = THING;
    sv.bpassign %wire42, %thing : i42

    sv.ifdef.procedural "FOO" {
      // CHECK-NEXT: `ifdef FOO
      %c1 = sv.verbatim.expr "\"THING\"" : () -> i1
      sv.fwrite "%d" (%c1) : i1
      // CHECK-NEXT: fwrite(32'h80000002, "%d", "THING");
      sv.fwrite "%d" (%c1) : i1
      // CHECK-NEXT: fwrite(32'h80000002, "%d", "THING");
      // CHECK-NEXT: `endif
    }

    // CHECK-NEXT: wire42 <= THING;
    sv.passign %wire42, %thing : i42

    // CHECK-NEXT: casez (val)
    sv.casez %val : i8
    // CHECK-NEXT: 8'b0000001?: begin
    case b0000001x: {
      // CHECK-NEXT: $fwrite(32'h80000002, "a");
      sv.fwrite "a"
      // CHECK-NEXT: $fwrite(32'h80000002, "b");
      sv.fwrite "b"
    } // CHECK-NEXT: end

    // CHECK-NEXT: 8'b000000?1:
    // CHECK-NOT: begin
    case b000000x1: {
      // CHECK-NEXT:  $fwrite(32'h80000002, "y");
      sv.fwrite "y"
    }  // implicit yield is ok.
    // CHECK-NEXT: default:
    // CHECK-NOT: begin
    default: {
      // CHECK-NEXT:  $fwrite(32'h80000002, "z");
      sv.fwrite "z"
    } // CHECK-NEXT: endcase

   // CHECK-NEXT: casez (cond)
   sv.casez %cond : i1
   // CHECK-NEXT: 1'b0:
     case b0: {
       // CHECK-NEXT: $fwrite(32'h80000002, "zero");
       sv.fwrite "zero"
     }
     // CHECK-NEXT: 1'b1:
     case b1: {
       // CHECK-NEXT: $fwrite(32'h80000002, "one");
       sv.fwrite "one"
     } // CHECK-NEXT: endcase
  }// CHECK-NEXT:   {{end // initial$}}

  sv.ifdef "VERILATOR"  {          // CHECK-NEXT: `ifdef VERILATOR
    sv.verbatim "`define Thing2"   // CHECK-NEXT:   `define Thing2
  } else  {                        // CHECK-NEXT: `else
    sv.verbatim "`define Thing1"   // CHECK-NEXT:   `define Thing1
  }                                // CHECK-NEXT: `endif

  %add = comb.add %val, %val : i8

  // CHECK-NEXT: `define STUFF "wire42 (val + val)"
  sv.verbatim "`define STUFF \"{{0}} ({{1}})\"" (%wire42, %add) : !rtl.inout<i42>, i8

  // CHECK-NEXT: `ifdef FOO
  sv.ifdef "FOO" {
    // CHECK-NEXT: wire {{.+}} = "THING";
    %c1 = sv.verbatim.expr "\"THING\"" : () -> i1

    // CHECK-NEXT: initial begin
    sv.initial {
      // CHECK-NEXT: fwrite(32'h80000002, "%d", {{.+}});
      sv.fwrite "%d" (%c1) : i1
      // CHECK-NEXT: fwrite(32'h80000002, "%d", {{.+}});
      sv.fwrite "%d" (%c1) : i1

    // CHECK-NEXT: end // initial
    }

  // CHECK-NEXT: `endif
  }
}

// CHECK-LABEL: module Aliasing(
// CHECK-NEXT:             inout [41:0] a, b, c
rtl.module @Aliasing(%a : !rtl.inout<i42>, %b : !rtl.inout<i42>,
                      %c : !rtl.inout<i42>) {

  // CHECK: alias a = b;
  sv.alias %a, %b     : !rtl.inout<i42>, !rtl.inout<i42>
  // CHECK: alias a = b = c;
  sv.alias %a, %b, %c : !rtl.inout<i42>, !rtl.inout<i42>, !rtl.inout<i42>
}

rtl.module @reg_0(%in4: i4, %in8: i8) -> (%a: i8, %b: i8) {
  // CHECK-LABEL: module reg_0(
  // CHECK-NEXT:   input  [3:0] in4,
  // CHECK-NEXT:   input  [7:0] in8,
  // CHECK-NEXT:   output [7:0] a, b);

  // CHECK-EMPTY:
  // CHECK-NEXT: reg [7:0]       myReg;
  %myReg = sv.reg : !rtl.inout<i8>

  // CHECK-NEXT: reg [41:0][7:0] myRegArray1;
  %myRegArray1 = sv.reg : !rtl.inout<array<42 x i8>>

  // CHECK-EMPTY:
  sv.connect %myReg, %in8 : i8        // CHECK-NEXT: assign myReg = in8;

  %subscript1 = sv.array_index_inout %myRegArray1[%in4] : !rtl.inout<array<42 x i8>>, i4
  sv.connect %subscript1, %in8 : i8   // CHECK-NEXT: assign myRegArray1[in4] = in8;

  %regout = sv.read_inout %myReg : !rtl.inout<i8>

  %subscript2 = sv.array_index_inout %myRegArray1[%in4] : !rtl.inout<array<42 x i8>>, i4
  %memout = sv.read_inout %subscript2 : !rtl.inout<i8>

  // CHECK-NEXT: assign a = myReg;
  // CHECK-NEXT: assign b = myRegArray1[in4];
  rtl.output %regout, %memout : i8, i8
}

// CHECK-LABEL: issue508
// https://github.com/llvm/circt/issues/508
rtl.module @issue508(%in1: i1, %in2: i1) {
  // CHECK: wire _T = in1 | in2;
  %clock = comb.or %in1, %in2 : i1 

  // CHECK-NEXT: always @(posedge _T) begin
  // CHECK-NEXT: end
  sv.always posedge %clock {
  }
}

// CHECK-LABEL: exprInlineTestIssue439
// https://github.com/llvm/circt/issues/439
rtl.module @exprInlineTestIssue439(%clk: i1) {
  // CHECK: always @(posedge clk) begin
  sv.always posedge %clk {
    %c = rtl.constant 0 : i32

    // CHECK: localparam      [31:0] _T = 32'h0;
    // CHECK: automatic logic [15:0] _T_0;
    // CHECK: _T_0 = _T[15:0];
    %e = comb.extract %c from 0 : (i32) -> i16
    %f = comb.add %e, %e : i16
    sv.fwrite "%d"(%f) : i16
    // CHECK: $fwrite(32'h80000002, "%d", _T_0 + _T_0);
    // CHECK: end // always @(posedge)
  }
}

// CHECK-LABEL: module issue439(
// https://github.com/llvm/circt/issues/439
rtl.module @issue439(%in1: i1, %in2: i1) {
  // CHECK: wire _T;
  // CHECK: wire _T_0 = in1 | in2;
  %clock = comb.or %in1, %in2 : i1

  // CHECK-NEXT: always @(posedge _T_0)
  sv.always posedge %clock {
    // CHECK-NEXT: _T <= in1;
    // CHECK-NEXT: _T <= in2;
    %merged = comb.merge %in1, %in2 : i1
    // CHECK-NEXT: $fwrite(32'h80000002, "Bye %x\n", _T);
    sv.fwrite "Bye %x\n"(%merged) : i1
  }
}

// CHECK-LABEL: module issue726(
// https://github.com/llvm/circt/issues/726
rtl.module @issue726(%in1: i1, %in2: i1) -> (%out: i1) {
  // CHECK: wire _T;
  // CHECK: assign _T = in1;
  // CHECK: assign _T = in2;
  %merged = comb.merge %in1, %in2 : i1

  // CHECK: assign out = _T;
  rtl.output %merged : i1
}

// https://github.com/llvm/circt/issues/595
// CHECK-LABEL: module issue595
rtl.module @issue595(%arr: !rtl.array<128xi1>) {
  // CHECK: wire [31:0] _T;
  // CHECK: wire _T_0 = _T == 32'h0;
  %c0_i32 = rtl.constant 0 : i32
  %c0_i7 = rtl.constant 0 : i7
  %c0_i6 = rtl.constant 0 : i6
  %0 = comb.icmp eq %3, %c0_i32 : i32

  sv.initial {
    // CHECK: assert(_T_0);
    sv.assert %0 : i1
  }

  // CHECK: wire [63:0] _T_1 = arr[7'h0+:64];
  // CHECK: assign _T = _T_1[6'h0+:32];
  %1 = rtl.array_slice %arr at %c0_i7 : (!rtl.array<128xi1>) -> !rtl.array<64xi1>
  %2 = rtl.array_slice %1 at %c0_i6 : (!rtl.array<64xi1>) -> !rtl.array<32xi1>
  %3 = rtl.bitcast %2 : (!rtl.array<32xi1>) -> i32
  rtl.output
}


// CHECK-LABEL: module issue595_variant1
rtl.module @issue595_variant1(%arr: !rtl.array<128xi1>) {
  // CHECK: wire [31:0] _T;
  // CHECK: wire _T_0 = |_T;
  %c0_i32 = rtl.constant 0 : i32
  %c0_i7 = rtl.constant 0 : i7
  %c0_i6 = rtl.constant 0 : i6
  %0 = comb.icmp ne %3, %c0_i32 : i32

  sv.initial {
    // CHECK: assert(_T_0);
    sv.assert %0 : i1
  }

  // CHECK: wire [63:0] _T_1 = arr[7'h0+:64];
  // CHECK: assign _T = _T_1[6'h0+:32];
  %1 = rtl.array_slice %arr at %c0_i7 : (!rtl.array<128xi1>) -> !rtl.array<64xi1>
  %2 = rtl.array_slice %1 at %c0_i6 : (!rtl.array<64xi1>) -> !rtl.array<32xi1>
  %3 = rtl.bitcast %2 : (!rtl.array<32xi1>) -> i32
  rtl.output
}

// CHECK-LABEL: module issue595_variant2_checkRedunctionAnd
rtl.module @issue595_variant2_checkRedunctionAnd(%arr: !rtl.array<128xi1>) {
  // CHECK: wire [31:0] _T;
  // CHECK: wire _T_0 = &_T;
  %c0_i32 = rtl.constant -1 : i32
  %c0_i7 = rtl.constant 0 : i7
  %c0_i6 = rtl.constant 0 : i6
  %0 = comb.icmp eq %3, %c0_i32 : i32

  sv.initial {
    // CHECK: assert(_T_0);
    sv.assert %0 : i1
  }

  // CHECK: wire [63:0] _T_1 = arr[7'h0+:64];
  // CHECK: assign _T = _T_1[6'h0+:32];
  %1 = rtl.array_slice %arr at %c0_i7 : (!rtl.array<128xi1>) -> !rtl.array<64xi1>
  %2 = rtl.array_slice %1 at %c0_i6 : (!rtl.array<64xi1>) -> !rtl.array<32xi1>
  %3 = rtl.bitcast %2 : (!rtl.array<32xi1>) -> i32
  rtl.output
}

// CHECK-LABEL: if_multi_line_expr1
rtl.module @if_multi_line_expr1(%clock: i1, %reset: i1, %really_long_port: i11) {
  %tmp6 = sv.reg  : !rtl.inout<i25>

  // CHECK:      if (reset)
  // CHECK-NEXT:   tmp6 <= 25'h0;
  // CHECK-NEXT: else begin
  // CHECK-NEXT:   automatic logic [24:0] _tmp;
  // CHECK-NEXT:   _tmp = {{..}}14{really_long_port[10]}}, really_long_port};
  // CHECK-NEXT:   tmp6 <= _tmp & 25'h3039;
  // CHECK-NEXT: end
  sv.alwaysff(posedge %clock)  {
    %0 = comb.sext %really_long_port : (i11) -> i25
  %c12345_i25 = rtl.constant 12345 : i25
    %1 = comb.and %0, %c12345_i25 : i25
    sv.passign %tmp6, %1 : i25
  }(syncreset : posedge %reset)  {
    %c0_i25 = rtl.constant 0 : i25
    sv.passign %tmp6, %c0_i25 : i25
  }
  rtl.output
}

// CHECK-LABEL: if_multi_line_expr2
rtl.module @if_multi_line_expr2(%clock: i1, %reset: i1, %really_long_port: i11) {
  %tmp6 = sv.reg  : !rtl.inout<i25>

  %c12345_i25 = rtl.constant 12345 : i25
  %0 = comb.sext %really_long_port : (i11) -> i25
  %1 = comb.and %0, %c12345_i25 : i25
  // CHECK:        wire [24:0] _tmp = {{..}}14{really_long_port[10]}}, really_long_port};
  // CHECK-NEXT:   wire [24:0] _T = _tmp & 25'h3039;

  // CHECK:      if (reset)
  // CHECK-NEXT:   tmp6 <= 25'h0;
  // CHECK-NEXT: else
  // CHECK-NEXT:   tmp6 <= _T;
  sv.alwaysff(posedge %clock)  {
    sv.passign %tmp6, %1 : i25
  }(syncreset : posedge %reset)  {
    %c0_i25 = rtl.constant 0 : i25
    sv.passign %tmp6, %c0_i25 : i25
  }
  rtl.output
}

// https://github.com/llvm/circt/issues/720
// CHECK-LABEL: module issue720(
rtl.module @issue720(%clock: i1, %arg1: i1, %arg2: i1, %arg3: i1) {

  // CHECK: always @(posedge clock) begin
  sv.always posedge %clock  {
    // CHECK:   automatic logic _T;

    // CHECK:   if (arg1)
    // CHECK:     $fatal;
    sv.if %arg1  {
      sv.fatal
    }

    // CHECK:   _T = arg1 & arg2;
    // CHECK:   if (_T)
    // CHECK:     $fatal;

    //this forces a common subexpression to be output out-of-line
    %610 = comb.and %arg1, %arg2 : i1
    %611 = comb.and %arg3, %610 : i1
    sv.if %610  {
      sv.fatal
    }

    // CHECK:   if (arg3 & _T)
    // CHECK:     $fatal;
    sv.if %611  {
      sv.fatal
    }
  } // CHECK: end // always @(posedge)

  rtl.output
}

// CHECK-LABEL: module issue720ifdef(
rtl.module @issue720ifdef(%clock: i1, %arg1: i1, %arg2: i1, %arg3: i1) {
  // CHECK: always @(posedge clock) begin
  sv.always posedge %clock  {
    // The variable for the ifdef block needs to be emitted at the top of the
    // always block since the ifdef is transparent to verilog.

    // CHECK:    automatic logic _T;
    // CHECK:    if (arg1)
    // CHECK:      $fatal;
    sv.if %arg1  {
      sv.fatal
    }

    // CHECK:    `ifdef FUN_AND_GAMES
     sv.ifdef.procedural "FUN_AND_GAMES" {
      // This forces a common subexpression to be output out-of-line
      // CHECK:      _T = arg1 & arg2;
      // CHECK:      if (_T)
      // CHECK:        $fatal;
      %610 = comb.and %arg1, %arg2 : i1
      sv.if %610  {
        sv.fatal
      }
      // CHECK:      if (arg3 & _T)
      // CHECK:        $fatal;
      %611 = comb.and %arg3, %610 : i1
     sv.if %611  {
        sv.fatal
      }
      // CHECK:    `endif
      // CHECK:  end // always @(posedge)
    }
  }
  rtl.output
}

// https://github.com/llvm/circt/issues/728

// CHECK-LABEL: module issue728(
rtl.module @issue728(%clock: i1, %a: i1, %b: i1)
attributes { argNames = ["clock", "asdfasdfasdfasdfafa", "gasfdasafwjhijjafija"] } {
  // CHECK:  always @(posedge clock) begin
  // CHECK:    automatic logic _tmp;
  // CHECK:    automatic logic _tmp_0;
  // CHECK:    $fwrite(32'h80000002, "force output");
  // CHECK:    _tmp = asdfasdfasdfasdfafa & gasfdasafwjhijjafija & asdfasdfasdfasdfafa;
  // CHECK:    _tmp_0 = gasfdasafwjhijjafija & asdfasdfasdfasdfafa & gasfdasafwjhijjafija;
  // CHECK:    if (_tmp & _tmp_0)
  // CHECK:      $fwrite(32'h80000002, "this cond is split");
  // CHECK:  end // always @(posedge)
  sv.always posedge %clock  {
     sv.fwrite "force output"
     %cond = comb.and %a, %b, %a, %b, %a, %b : i1
     sv.if %cond  {
       sv.fwrite "this cond is split"
     }
  }
  rtl.output 
}

// CHECK-LABEL: module issue728ifdef(
rtl.module @issue728ifdef(%clock: i1, %a: i1, %b: i1)
  attributes { argNames = ["clock", "asdfasdfasdfasdfafa", "gasfdasafwjhijjafija"] } {
  // CHECK: always @(posedge clock) begin
  // CHECK:      automatic logic _tmp;
  // CHECK:      automatic logic _tmp_0;
  // CHECK:    $fwrite(32'h80000002, "force output");
  // CHECK:    `ifdef FUN_AND_GAMES
  // CHECK:      _tmp = asdfasdfasdfasdfafa & gasfdasafwjhijjafija & asdfasdfasdfasdfafa;
  // CHECK:      _tmp_0 = gasfdasafwjhijjafija & asdfasdfasdfasdfafa & gasfdasafwjhijjafija;
  // CHECK:      if (_tmp & _tmp_0)
  // CHECK:        $fwrite(32'h80000002, "this cond is split");
  // CHECK:    `endif
  // CHECK: end // always @(posedge)
  sv.always posedge %clock  {
     sv.fwrite "force output"
     sv.ifdef.procedural "FUN_AND_GAMES" {
       %cond = comb.and %a, %b, %a, %b, %a, %b : i1
       sv.if %cond  {
         sv.fwrite "this cond is split"
       }
     }
  }
}

// CHECK-LABEL: module alwayscombTest(
rtl.module @alwayscombTest(%a: i1) -> (%x: i1) {
  // CHECK: wire combWire;
  %combWire = sv.wire : !rtl.inout<i1>
  // CHECK: always_comb
  sv.alwayscomb {
    // CHECK-NEXT: combWire <= a
    sv.passign %combWire, %a : i1
  }

  // CHECK: assign x = combWire;
  %out = sv.read_inout %combWire : !rtl.inout<i1>
  rtl.output %out : i1
}


// https://github.com/llvm/circt/issues/838
// CHECK-LABEL: module inlineProceduralWiresWithLongNames(
rtl.module @inlineProceduralWiresWithLongNames(%clock: i1, %in: i1) {
  %aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = sv.wire  : !rtl.inout<i1>
  %0 = sv.read_inout %aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa : !rtl.inout<i1>
  %bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb = sv.wire  : !rtl.inout<i1>
  %1 = sv.read_inout %bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb : !rtl.inout<i1>
  %r = sv.reg  : !rtl.inout<uarray<1xi1>>
  %s = sv.reg  : !rtl.inout<uarray<1xi1>>
  %2 = sv.array_index_inout %r[%0] : !rtl.inout<uarray<1xi1>>, i1
  %3 = sv.array_index_inout %s[%1] : !rtl.inout<uarray<1xi1>>, i1
  // CHECK: wire _T = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;
  // CHECK: wire _T_0 = bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;
  // CHECK: always_ff
  sv.alwaysff(posedge %clock)  {
    // CHECK-NEXT: r[_T] <= in;
    sv.passign %2, %in : i1
    // CHECK-NEXT: s[_T_0] <= in;
    sv.passign %3, %in : i1
  }
}

// https://github.com/llvm/circt/issues/859
// CHECK-LABEL: module oooReg(
rtl.module @oooReg(%in: i1) -> (%result: i1) {
  // CHECK: wire abc;
  %0 = sv.read_inout %abc : !rtl.inout<i1>

  // CHECK: assign abc = in;
  sv.connect %abc, %in : i1
  %abc = sv.wire  : !rtl.inout<i1>

  // CHECK: assign result = abc;
  rtl.output %0 : i1
}

// https://github.com/llvm/circt/issues/865
// CHECK-LABEL: module ifdef_beginend(
rtl.module @ifdef_beginend(%clock: i1, %cond: i1, %val: i8) {
  // CHECK: always @(posedge clock) begin
  sv.always posedge %clock  {
    // CHECK-NEXT: `ifndef SYNTHESIS
    sv.ifdef.procedural "SYNTHESIS"  {
    } // CHECK-NEXT: `endif
  } // CHECK-NEXT: end
} // CHECK-NEXT: endmodule

// https://github.com/llvm/circt/issues/884
// CHECK-LABEL: module ConstResetValueMustBeInlined(
rtl.module @ConstResetValueMustBeInlined(%clock: i1, %reset: i1, %d: i42) -> (%q: i42) {
  %c0_i42 = rtl.constant 0 : i42
  %tmp = sv.reg : !rtl.inout<i42>
  // CHECK:      localparam [41:0] _T = 42'h0;
  // CHECK-NEXT: always_ff @(posedge clock or posedge reset) begin
  // CHECK-NEXT:   if (reset)
  // CHECK-NEXT:     tmp <= _T;
  sv.alwaysff(posedge %clock) {
    sv.passign %tmp, %d : i42
  } (asyncreset : posedge %reset)  {
    sv.passign %tmp, %c0_i42 : i42
  }
  %1 = sv.read_inout %tmp : !rtl.inout<i42>
  rtl.output %1 : i42
}

// CHECK-LABEL: module OutOfLineConstantsInAlwaysSensitivity
rtl.module @OutOfLineConstantsInAlwaysSensitivity() {
  // CHECK-NEXT: localparam _T = 1'h0;
  // CHECK-NEXT: always_ff @(posedge _T)
  %clock = rtl.constant 0 : i1
  sv.alwaysff(posedge %clock) {}
}

// CHECK-LABEL: module TooLongConstExpr
rtl.module @TooLongConstExpr() {
  %myreg = sv.reg : !rtl.inout<i4200>
  // CHECK: always @*
  sv.always {
    // CHECK-NEXT: localparam [4199:0] _tmp = 4200'h
    // CHECK-NEXT: myreg <= _tmp + _tmp;
    %0 = rtl.constant 15894191981981165163143546843135416146464164161464654561818646486465164684484 : i4200
    %1 = comb.add %0, %0 : i4200
    sv.passign %myreg, %1 : i4200
  }
}

// Constants defined before use should be emitted in-place.
// CHECK-LABEL: module ConstantDefBeforeUse
rtl.module @ConstantDefBeforeUse() {
  %myreg = sv.reg : !rtl.inout<i32>
  // CHECK:      localparam [31:0] _T = 32'h2A;
  // CHECK-NEXT: always @*
  // CHECK-NEXT:   myreg <= _T
  %0 = rtl.constant 42 : i32
  sv.always {
    sv.passign %myreg, %0 : i32
  }
}

// Constants defined after use in non-procedural regions should be moved to the
// top of the block.
// CHECK-LABEL: module ConstantDefAfterUse
rtl.module @ConstantDefAfterUse() {
  %myreg = sv.reg : !rtl.inout<i32>
  // CHECK:      localparam [31:0] _T = 32'h2A;
  // CHECK-NEXT: always @*
  // CHECK-NEXT:   myreg <= _T
  sv.always {
    sv.passign %myreg, %0 : i32
  }
  %0 = rtl.constant 42 : i32
}

// Constants defined in a procedural block with users in a different block
// should be emitted at the top of their defining block.
// CHECK-LABEL: module ConstantEmissionAtTopOfBlock
rtl.module @ConstantEmissionAtTopOfBlock() {
  %myreg = sv.reg : !rtl.inout<i32>
  // CHECK:      always @* begin
  // CHECK-NEXT:   localparam [31:0] _T = 32'h2A;
  // CHECK:          myreg <= _T;
  sv.always {
    %0 = rtl.constant 42 : i32
    %1 = rtl.constant 1 : i1
    sv.if %1 {
      sv.passign %myreg, %0 : i32
    }
  }
}

//CHECK-LABEL: bind ConstantDefAfterUse ConstantEmissionAtTopOfBlock foobar_inst (.*)
sv.bind "foobar_inst" @ConstantDefAfterUse @ConstantEmissionAtTopOfBlock

// CHECK-LABEL: module extInst
rtl.module.extern @extInst(%_h: i1, %_i: i1, %_j: i1, %_k: i1) -> ()

// CHECK-LABEL: module remoteInstDut
rtl.module @remoteInstDut(%i: i1, %j: i1) -> () {
  %mywire = sv.wire : !rtl.inout<i1>
  %mywire_rd = sv.read_inout %mywire : !rtl.inout<i1>
  %myreg = sv.reg : !rtl.inout<i1>
  %myreg_rd = sv.read_inout %myreg : !rtl.inout<i1>
  %0 = rtl.constant 1 : i1
  rtl.instance "a1" sym @bindInst @extInst(%mywire_rd, %myreg_rd, %j, %0) {emitAsBind=1}: (i1, i1, i1, i1) -> ()
  rtl.instance "a2" sym @bindInst @extInst(%mywire_rd, %myreg_rd, %j, %0) : (i1, i1, i1, i1) -> ()
// CHECK: wire extInst_a1__k
// CHECK-NEXT: extInst a2
}

sv.bind.explicit @bindInst
//CHECK-LABEL: bind remoteInstDut extInst a1 (
//CHECK-NEXT:  ._h (mywire),
//CHECK-NEXT:  ._i (myreg),
//CHECK-NEXT:  ._j (j),
//CHECK-NEXT:  ._k (extInst_a1__k)
//CHECK-NEXT:);

rtl.module @bindInMod() -> () {
  sv.bind.explicit @bindInst
  sv.bind "foobar2_inst"  @ConstantDefAfterUse @ConstantEmissionAtTopOfBlock
}
//CHECK-LABEL: bindInMod
//CHECK-NEXT:  bind remoteInstDut extInst a1 (
//CHECK-NEXT:    ._h (mywire),
//CHECK-NEXT:    ._i (myreg),
//CHECK-NEXT:    ._j (j),
//CHECK-NEXT:    ._k (extInst_a1__k)
//CHECK-NEXT:  );
//CHECK-NEXT:  bind ConstantDefAfterUse ConstantEmissionAtTopOfBlock foobar2_inst (.*);
