// REQUIRES: capnp
// RUN: circt-translate %s -export-esi-capnp -verify-diagnostics | FileCheck --check-prefix=CAPNP %s

!DataPkt = type !esi.struct<DataPkt, encrypted: i1, compressionLevel: ui4, data: !rtl.array<32 x i8>>
!pktChan = type !esi.channel<!DataPkt>

rtl.module.extern @Compressor(!pktChan) -> (!pktChan { rtl.name = "x"})

rtl.module @top(%clk:i1, %rstn:i1) -> () {
  %compressedData = rtl.instance "compressor" @Compressor(%inputData) : (!pktChan) -> !pktChan
  %inputData = esi.cosim %clk, %rstn, %compressedData, 1 {name="Compressor"} : !pktChan -> !pktChan
}

// CAPNP:      struct DataPkt
// CAPNP-NEXT:   encrypted        @0 :Bool;
// CAPNP-NEXT:   compressionLevel @1 :UInt8;
// CAPNP-NEXT:   data             @2 :List(UInt8);
