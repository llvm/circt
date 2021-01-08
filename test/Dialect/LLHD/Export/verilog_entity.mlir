//RUN: circt-translate --llhd-to-verilog %s | FileCheck %s

// CHECK: module _empty;
llhd.entity @empty () -> () {
// CHECK-NEXT: endmodule
}

// CHECK-NEXT: module _onlyinput(input [63:0] _{{.*}});
llhd.entity @onlyinput (%arg0 : !llhd.sig<i64>) -> () {
// CHECK-NEXT: endmodule
}

// CHECK-NEXT: module _onlyoutput(output [63:0] _{{.*}});
llhd.entity @onlyoutput () -> (%out0 : !llhd.sig<i64>) {
// CHECK-NEXT: endmodule
}

// CHECK-NEXT: module _inputandoutput(input [63:0] _{{.*}}, input [31:0] _{{.*}}, output [7:0] _{{.*}}, output [15:0] _{{.*}});
llhd.entity @inputandoutput (%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i32>) -> (%out0 : !llhd.sig<i8>, %out1 : !llhd.sig<i16>) {
// CHECK-NEXT: endmodule
}

// CHECK-NEXT: module _check_inst(input [63:0] _[[ARG0:.*]], input [31:0] _[[ARG1:.*]], output [7:0] _[[OUT0:.*]], output [15:0] _[[OUT1:.*]]);
llhd.entity @check_inst (%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i32>) -> (%out0 : !llhd.sig<i8>, %out1 : !llhd.sig<i16>) {
  // CHECK-NEXT: _empty inst_[[INST0:.*]];
  llhd.inst "empty" @empty () -> () : () -> ()
  // CHECK-NEXT: _onlyinput inst_[[INST1:.*]] (_[[ARG0]]);
  llhd.inst "input" @onlyinput (%arg0) -> () : (!llhd.sig<i64>) -> ()
  // CHECK-NEXT: _onlyoutput inst_[[INST2:.*]] (_[[ARG0]]);
  llhd.inst "output" @onlyoutput () -> (%arg0) : () -> (!llhd.sig<i64>)
  // CHECK-NEXT: _inputandoutput inst_[[INST3:.*]] (_[[ARG0]], _[[ARG1]], _[[OUT0]], _[[OUT1]]);
  llhd.inst "in-out" @inputandoutput (%arg0, %arg1) -> (%out0, %out1) : (!llhd.sig<i64>, !llhd.sig<i32>) -> (!llhd.sig<i8>, !llhd.sig<i16>)
  // CHECK-NEXT: endmodule
}
