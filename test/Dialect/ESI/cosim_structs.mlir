// REQUIRES: capnp
// RUN: circt-translate %s -export-esi-capnp -verify-diagnostics | FileCheck --check-prefix=CAPNP %s
// RUN: circt-opt %s --lower-esi-ports --lower-esi-to-rtl -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck --check-prefix=COSIM %s
// RUN: circt-opt %s --lower-esi-ports --lower-esi-to-rtl | circt-translate --export-verilog | FileCheck --check-prefix=SV %s

!DataPkt = type !rtl.struct<encrypted: i1, compressionLevel: ui4, blob: !rtl.array<32 x i8>>
!pktChan = type !esi.channel<!DataPkt>

rtl.module.extern @Compressor(%in: !esi.channel<i1>) -> (!pktChan { rtl.name = "x"})

rtl.module @top(%clk:i1, %rstn:i1) -> () {
  %compressedData = rtl.instance "compressor" @Compressor(%inputData) : (!esi.channel<i1>) -> !pktChan
  %inputData = esi.cosim %clk, %rstn, %compressedData, 1 {name="Compressor"} : !pktChan -> !esi.channel<i1>
}

// CAPNP:      struct Struct13922113893393513056
// CAPNP-NEXT:   encrypted        @0 :Bool;
// CAPNP-NEXT:   compressionLevel @1 :UInt8;
// CAPNP-NEXT:   blob             @2 :List(UInt8);

// COSIM: rtl.instance "Compressor" @Cosim_Endpoint(%clk, %rstn, %{{.+}}, %{{.+}}, %{{.+}}) {parameters = {ENDPOINT_ID = 1 : i32, RECV_TYPE_ID = {{[0-9]+}} : ui64, RECV_TYPE_SIZE_BITS = 128 : i32, SEND_TYPE_ID = {{[0-9]+}} : ui64, SEND_TYPE_SIZE_BITS = 448 : i32}}

// Test only a single, critical line in the systemverilog output. Anything else would be too fragile.
// SV: wire [31:0][7:0] [[IF1:.+]] = ({{.+}}.data).blob;
// SV: .DataIn       ({[[IF1]][5'h0], [[IF1]][5'h1], [[IF1]][5'h2], [[IF1]][5'h3], [[IF1]][5'h4], [[IF1]][5'h5], [[IF1]][5'h6], [[IF1]][5'h7], [[IF1]][5'h8], [[IF1]][5'h9], [[IF1]][5'hA], [[IF1]][5'hB], [[IF1]][5'hC], [[IF1]][5'hD], [[IF1]][5'hE], [[IF1]][5'hF], [[IF1]][5'h10], [[IF1]][5'h11], [[IF1]][5'h12], [[IF1]][5'h13], [[IF1]][5'h14], [[IF1]][5'h15], [[IF1]][5'h16], [[IF1]][5'h17], [[IF1]][5'h18], [[IF1]][5'h19], [[IF1]][5'h1A], [[IF1]][5'h1B], [[IF1]][5'h1C], [[IF1]][5'h1D], [[IF1]][5'h1E], [[IF1]][5'h1F], {29'h20, 3'h2, 30'h0, 2'h1}, 52'h0, (_T_0.data).compressionLevel, 7'h0, (_T_0.data).encrypted, {16'h1, 16'h1, 30'h0, 2'h0}})
