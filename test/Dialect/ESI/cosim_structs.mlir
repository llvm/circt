// REQUIRES: capnp
// RUN: circt-translate %s -export-esi-capnp -verify-diagnostics | FileCheck --check-prefix=CAPNP %s
// RUN: circt-opt %s --lower-esi-ports --lower-esi-to-rtl -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck --check-prefix=COSIM %s

!DataPkt = type !rtl.struct<encrypted: i1, compressionLevel: ui4, blob: !rtl.array<32 x i8>>
!pktChan = type !esi.channel<!DataPkt>

rtl.module.extern @Compressor(%in: !esi.channel<i1>) -> (%x: !pktChan)

rtl.module @top(%clk:i1, %rstn:i1) -> () {
  %compressedData = rtl.instance "compressor" @Compressor(%inputData) : (!esi.channel<i1>) -> !pktChan
  %inputData = esi.cosim %clk, %rstn, %compressedData, 1 {name="Compressor"} : !pktChan -> !esi.channel<i1>
}

// CAPNP:      struct Struct{{.+}}
// CAPNP-NEXT:   encrypted        @0 :Bool;
// CAPNP-NEXT:   compressionLevel @1 :UInt8;
// CAPNP-NEXT:   blob             @2 :List(UInt8);

// COSIM: rtl.instance "encodeStruct{{.+}}Inst" @encodeStruct{{.+}}(%clk, %6, %7) : (i1, i1, !rtl.struct<encrypted: i1, compressionLevel: ui4, blob: !rtl.array<32xi8>>) -> !rtl.array<448xi1>
// COSIM: rtl.instance "Compressor" @Cosim_Endpoint(%clk, %rstn, %{{.+}}, %{{.+}}, %{{.+}}) {parameters = {ENDPOINT_ID = 1 : i32, RECV_TYPE_ID = {{[0-9]+}} : ui64, RECV_TYPE_SIZE_BITS = 128 : i32, SEND_TYPE_ID = {{[0-9]+}} : ui64, SEND_TYPE_SIZE_BITS = 448 : i32}}
// COSIM: rtl.module @encode{{.+}}(%clk: i1, %valid: i1, %unencodedInput: !rtl.struct<encrypted: i1, compressionLevel: ui4, blob: !rtl.array<32xi8>>) -> (%encoded: !rtl.array<448xi1>)
