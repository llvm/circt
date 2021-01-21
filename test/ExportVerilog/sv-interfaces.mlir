// RUN: circt-translate %s -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

module {
  // CHECK-LABEL: interface data_vr;
  // CHECK:         logic [31:0] data;
  // CHECK:         logic valid;
  // CHECK:         logic ready;
  // CHECK:         modport data_in(input data, input valid, output ready);
  // CHECK:         modport data_out(output data, output valid, input ready);
  // CHECK:       endinterface
  // CHECK-EMPTY:
  sv.interface @data_vr {
    sv.interface.signal @data : i32
    sv.interface.signal @valid : i1
    sv.interface.signal @ready : i1
    sv.interface.signal @arrayData : !rtl.array<8xi8>
    sv.interface.modport @data_in ("input" @data, "input" @valid, "output" @ready)
    sv.interface.modport @data_out ("output" @data, "output" @valid, "input" @ready)
  }

  rtl.externmodule @Rcvr (%m: !sv.modport<@data_vr::@data_in>)

  // CHECK-LABEL: module Top
  rtl.module @Top (%clk: i1) {
    // CHECK: data_vr [[IFACE:.+]]();{{.*}}//{{.+}}
    %iface = sv.interface.instance : !sv.interface<@data_vr>

    // CHECK-EMPTY:
    %ifaceInPort = sv.modport.get %iface @data_in :
      !sv.interface<@data_vr> -> !sv.modport<@data_vr::@data_in>

    // CHECK: Rcvr rcvr1 ({{.*}}//{{.+}}
    // CHECK:   .m ([[IFACE]].data_in){{.*}}//{{.+}}
    // CHECK: );
    rtl.instance "rcvr1" @Rcvr(%ifaceInPort) : (!sv.modport<@data_vr::@data_in>) -> ()

    // CHECK: Rcvr rcvr2 ({{.*}}//{{.+}}
    // CHECK:   .m ([[IFACE]].data_in){{.*}}//{{.+}}
    // CHECK: );
    rtl.instance "rcvr2" @Rcvr(%ifaceInPort) : (!sv.modport<@data_vr::@data_in>) -> ()

    %c1 = rtl.constant (1 : i1) : i1
    // CHECK: assign _T.valid = 1'h1;
    sv.interface.signal.assign %iface(@data_vr::@valid) = %c1 : i1

    sv.always posedge %clk {
      %validValue = sv.interface.signal.read %iface(@data_vr::@valid) : i1
      // CHECK: $fwrite(32'h80000002, "valid: %d\n", _T.valid);
      sv.fwrite "valid: %d\n" (%validValue) : i1
      // CHECK: assert(_T.valid);
      sv.assert %validValue : i1
    }
  }
}
