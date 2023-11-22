// RUN: circt-opt %s --esi-appid-hier=top=main | FileCheck %s

hw.module @main(in %clk : !seq.clock, in %rst : i1) {
  hw.instance "Top" sym @Top @Top() -> () 
  esi.manifest.service_impl #esi.appid<"cosim"[0]> by "cosim" with {} {
    esi.manifest.impl_conn [#esi.appid<"loopback_inout"[0]>] req <@HostComms::@req_resp>(!esi.bundle<[!esi.channel<i16> from "resp", !esi.channel<i24> to "req"]>) with {channel_assignments = {req = "loopback_inout[0].req", resp = "loopback_inout[0].resp"}}
  }
}

hw.module @Top() {
  hw.instance "LoopbackInOutAdd7" sym @LoopbackInOutAdd7 @LoopbackInOutAdd7() -> ()
  hw.instance "Bar" sym @Bar @Bar() -> () {esi.appid=#esi.appid<"bar"[0]>}
}

hw.module @LoopbackInOutAdd7() {
  esi.manifest.req #esi.appid<"loopback_inout"[0]>, <@HostComms::@req_resp>, toServer, !esi.bundle<[!esi.channel<i16> to "resp", !esi.channel<i24> from "req"]>
}

hw.module @Bar () {}

// CHECK-LABEL: esi.manifest.hier_root @main {
// CHECK-NEXT:    esi.manifest.req #esi.appid<"loopback_inout"[0]>, <@HostComms::@req_resp>, toServer, !esi.bundle<[!esi.channel<i16> to "resp", !esi.channel<i24> from "req"]>
// CHECK-NEXT:    esi.manifest.hier_node #esi.appid<"bar"[0]> mod @Bar {
// CHECK-NEXT:    }
// CHECK-NEXT:    esi.manifest.service_impl #esi.appid<"cosim"[0]> by "cosim" with {} {
// CHECK-NEXT:      esi.manifest.impl_conn [#esi.appid<"loopback_inout"[0]>] req <@HostComms::@req_resp>(!esi.bundle<[!esi.channel<i16> from "resp", !esi.channel<i24> to "req"]>) with {channel_assignments = {req = "loopback_inout[0].req", resp = "loopback_inout[0].resp"}}
// CHECK-NEXT:    }
// CHECK-NEXT:  }
