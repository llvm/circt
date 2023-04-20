// RUN: circt-opt %s --esi-emit-cpp-cosim-api="to-stderr=true" -o %t 2>&1 | FileCheck %s

esi.service.decl @HostComms {
  esi.service.to_server @SendStruct : !esi.channel<i8>
  esi.service.to_client @Recv : !esi.channel<i8>
  esi.service.inout @ReqResp : !esi.channel<i8> -> !esi.channel<i16>
}

esi.service.decl @BSP {
  esi.service.to_client @Recv : !esi.channel<i8>
  esi.service.to_server @Send : !esi.channel<i8>
}

hw.module @Top(%clk: i1, %rst: i1) {
  %0 = esi.null : !esi.channel<i1>
  %1 = esi.cosim %clk, %rst, %0, "m1.loopback_tohw" : !esi.channel<i1> -> !esi.channel<i8>
  %2 = esi.cosim %clk, %rst, %m1.loopback_fromhw, "m1.loopback_fromhw" : !esi.channel<i8> -> !esi.channel<i1>
  esi.service.hierarchy.metadata path [] implementing @BSP impl as "cosim" clients [{client_name = ["m1", "loopback_tohw"], port = #hw.innerNameRef<@HostComms::@Recv>, to_client_type = !esi.channel<i8>}, {client_name = ["m1", "loopback_fromhw"], port = #hw.innerNameRef<@HostComms::@Send>, to_server_type = !esi.channel<i8>}]
  %m1.loopback_fromhw = hw.instance "m1" @Loopback(clk: %clk: i1, loopback_tohw: %1: !esi.channel<i8>) -> (loopback_fromhw: !esi.channel<i8>)
  hw.output
}

hw.module @Loopback(%clk: i1, %loopback_tohw: !esi.channel<i8>) -> (loopback_fromhw: !esi.channel<i8>) {
  hw.output %loopback_tohw : !esi.channel<i8>
}