// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

esi.service.decl @HostComms {
  esi.service.to_server @Send : !esi.channel<!esi.any>
  esi.service.to_client @Recv : !esi.channel<i8>
}

hw.module @Top (%clk: i1) -> () {
  esi.service.instance "topComms" @HostComms (%clk) : (i1) -> ()
  hw.instance "m1" @Loopback (clk: %clk: i1) -> ()
}

hw.module @Loopback (%clk: i1) -> () {
  %dataIn = esi.service.req.to_client @HostComms ("loopback_tohw") : !esi.channel<i8>
  esi.service.req.to_server %dataIn -> @HostComms ("loopback_fromhw") : !esi.channel<i8>
}
