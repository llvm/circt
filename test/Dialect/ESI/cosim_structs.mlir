// REQUIRES: capnp
// RUN: circt-translate %s -export-esi-capnp -verify-diagnostics | FileCheck --check-prefix=CAPNP %s
// RUN: circt-opt %s --lower-esi-ports --lower-esi-to-hw -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck --check-prefix=COSIM %s

!DataPkt = type !hw.struct<encrypted: i1, compressionLevel: ui4, blob: !hw.array<32 x i8>>
!pktChan = type !esi.channel<!DataPkt>

hw.module.extern @Compressor(%in: !esi.channel<i1>) -> (x: !pktChan)

hw.module @top(%clk:i1, %rstn:i1) -> () {
  %compressedData = hw.instance "compressor" @Compressor(in: %inputData: !esi.channel<i1>) -> (x: !pktChan)
  %inputData = esi.cosim %clk, %rstn, %compressedData, 1 {name="Compressor"} : !pktChan -> !esi.channel<i1>
}

// CAPNP:      struct Struct{{.+}}
// CAPNP-NEXT:   encrypted        @0 :Bool;
// CAPNP-NEXT:   compressionLevel @1 :UInt8;
// CAPNP-NEXT:   blob             @2 :List(UInt8);

// COSIM: hw.instance "encodeStruct{{.+}}Inst" @encodeStruct{{.+}}(clk: %clk: i1, valid: %6: i1, unencodedInput: %7: !hw.struct<encrypted: i1, compressionLevel: ui4, blob: !hw.array<32xi8>>) -> (encoded: !hw.array<448xi1>)
// COSIM: hw.instance "Compressor" @Cosim_Endpoint(clk: %clk: i1, rstn: %rstn: i1, {{.+}}, {{.+}}, {{.+}}) -> ({{.+}}) {parameters = {ENDPOINT_ID = 1 : i32, RECV_TYPE_ID = {{[0-9]+}} : ui64, RECV_TYPE_SIZE_BITS = 128 : i32, SEND_TYPE_ID = {{[0-9]+}} : ui64, SEND_TYPE_SIZE_BITS = 448 : i32}}
// COSIM: hw.module @encode{{.+}}(%clk: i1, %valid: i1, %unencodedInput: !hw.struct<encrypted: i1, compressionLevel: ui4, blob: !hw.array<32xi8>>) -> (encoded: !hw.array<448xi1>)
