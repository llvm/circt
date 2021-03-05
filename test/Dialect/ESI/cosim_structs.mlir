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
// SV: wire [31:0][7:0] [[BLOB:.+]] = [[IF1:.+]].data.blob;
// SV: .DataIn       ({[[BLOB]][5'h0], [[BLOB]][5'h1], [[BLOB]][5'h2], [[BLOB]][5'h3], [[BLOB]][5'h4], [[BLOB]][5'h5], [[BLOB]][5'h6], [[BLOB]][5'h7], [[BLOB]][5'h8], [[BLOB]][5'h9], [[BLOB]][5'hA], [[BLOB]][5'hB], [[BLOB]][5'hC], [[BLOB]][5'hD], [[BLOB]][5'hE], [[BLOB]][5'hF], [[BLOB]][5'h10], [[BLOB]][5'h11], [[BLOB]][5'h12], [[BLOB]][5'h13], [[BLOB]][5'h14], [[BLOB]][5'h15], [[BLOB]][5'h16], [[BLOB]][5'h17], [[BLOB]][5'h18], [[BLOB]][5'h19], [[BLOB]][5'h1A], [[BLOB]][5'h1B], [[BLOB]][5'h1C], [[BLOB]][5'h1D], [[BLOB]][5'h1E], [[BLOB]][5'h1F], {29'h20, 3'h2, 30'h0, 2'h1}, 52'h0, [[IF1]].data.compressionLevel, 7'h0, [[IF1]].data.encrypted, {16'h1, 16'h1, 30'h0, 2'h0}})
