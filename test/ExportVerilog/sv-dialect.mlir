// RUN: circt-translate %s -emit-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

// CHECK-LABEL: interface myinterface;
// CHECK:         logic [31:0] data;
// CHECK:         modport input_port(input data);
// CHECK:         modport output_port(output data);
// CHECK:       endinterface
// CHECK-EMPTY:
sv.interface @myinterface {
  sv.interface.signal @data : i32
  sv.interface.modport @input_port ("input" @data)
  sv.interface.modport @output_port ("output" @data)
}

// CHECK-LABEL: interface handshake_example;
// CHECK:         logic [31:0] data;
// CHECK:         logic valid;
// CHECK:         logic ready;
// CHECK:         modport dataflow_in(input data, input valid, output ready);
// CHECK:         modport dataflow_out(output data, output valid, input ready);
// CHECK:       endinterface
// CHECK-EMPTY:
sv.interface @handshake_example {
  sv.interface.signal @data : i32
  sv.interface.signal @valid : i1
  sv.interface.signal @ready : i1
  sv.interface.modport @dataflow_in ("input" @data, "input" @valid, "output" @ready)
  sv.interface.modport @dataflow_out ("output" @data, "output" @valid, "input" @ready)
}

// CHECK-LABEL: module M1(
rtl.module @M1(%clock : i1, %cond : i1, %val : i8) {

  // CHECK:      always @(posedge clock) begin
  // CHECK-NEXT:   #ifndef SYNTHESIS
  // CHECK-NEXT:     if (PRINTF_COND_ & cond)
  // CHECK-NEXT:       $fwrite(32'h80000002, "Hi\n");
  // CHECK-NEXT:   #endif
  // CHECK-NEXT: end // always @(posedge)
  sv.alwaysat_posedge %clock {
    sv.ifdef "!SYNTHESIS" {
      %tmp = sv.textual_value "PRINTF_COND_" : i1
      %tmp2 = rtl.and %tmp, %cond : i1
      sv.if %tmp2 {
        sv.fwrite "Hi\n"
      }
    }
  }

  // CHECK-NEXT:   if (cond) begin
  sv.if %cond {
    // CHECK-NEXT:     $fwrite(32'h80000002, "Hi\n");
    sv.fwrite "Hi\n"

    // CHECK-NEXT:     $fwrite(32'h80000002, "Bye %x\n", val + val);
    %tmp = rtl.add %val, %val : i8
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
    // CHECK-NEXT:   {{end$}}
  }
}

// CHECK-LABEL: module Aliasing(
// CHECK-NEXT:             inout [41:0] a, b, c);
rtl.module @Aliasing(%a : !rtl.inout<i42>, %b : !rtl.inout<i42>,
                      %c : !rtl.inout<i42>) {

  // CHECK: alias a = b;
  sv.alias %a, %b     : !rtl.inout<i42>, !rtl.inout<i42>
  // CHECK: alias a = b = c;
  sv.alias %a, %b, %c : !rtl.inout<i42>, !rtl.inout<i42>, !rtl.inout<i42>
}


