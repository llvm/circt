// RUN: circt-opt --arc-resolve-xmr %s | FileCheck %s

module {
  hw.hierpath @bindPath [@Host::@src]

  // CHECK-LABEL: hw.module @Payload
  // CHECK-SAME: in %xmr_capture_src_{{[0-9]+}} : i8
  // CHECK: hw.output %{{.+}} : i8
  hw.module @Payload(out o : i8) {
    %x = sv.xmr.ref @bindPath : !hw.inout<i8>
    %r = sv.read_inout %x : !hw.inout<i8>
    hw.output %r : i8
  }

  // CHECK-LABEL: hw.module @Host
  // CHECK-SAME: in %src_in : i8
  // CHECK-SAME: out out : i8
  // CHECK-NEXT: %[[P:.+]] = hw.instance "payload" sym @payload @Payload(xmr_capture_src_{{[0-9]+}}: %src_in: i8) -> (o: i8) {doNotPrint}
  // CHECK-NEXT: hw.output %[[P]] : i8
  hw.module @Host(in %src_in : i8 {hw.exportPort = #hw<innerSym@src>}, out out : i8) {
    %p = hw.instance "payload" sym @payload @Payload() -> (o: i8) {doNotPrint}
    hw.output %p : i8
  }

  sv.bind <@Host::@payload>
}
