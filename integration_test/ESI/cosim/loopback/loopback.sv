// PY: import loopback as test
// PY: rpc = test.LoopbackTester(rpcschemapath, simhostport)
// PY: rpc.test_list()
// PY: rpc.test_open_close()
// PY: rpc.write_read_many(5)

import Cosim_DpiPkg::*;

module Top(
  input clk, rstn);

  wire         TestEP_DataOutValid;     // <stdin>:6:66
  wire [191:0] TestEP_DataOut;  // <stdin>:6:66
  wire         TestEP_DataInReady;      // <stdin>:6:66

  wire [191:0] _T = /*cast(bit[191:0])*/192'h0; // <stdin>:4:16, :5:10
  Cosim_Endpoint #(.ENDPOINT_ID(1), .RECVTYPE_SIZE_BITS(192), .RECV_TYPE_ID(-8151307203261699935), .SEND_TYPE_ID(-8151307203261699935), .SEND_TYPE_SIZE_BITS(192)) TestEP (  // <stdin>:6:66
    .clk (clk),
    .rstn (rstn),
    .DataOutReady (TestEP_DataInReady),
    .DataInValid (TestEP_DataInReady),
    .DataIn (_T),
    .DataOutValid (TestEP_DataOutValid),
    .DataOut (TestEP_DataOut),
    .DataInReady (TestEP_DataInReady)
  );
  wire [7:0] _T_0 = 8'h80;      // <stdin>:7:17
  wire [63:0] _T_1 = /*cast(bit[63:0])*/TestEP_DataOut[_T_0+:64];       // <stdin>:8:10, :9:10
  assert(_T_1 == 64'h80);       // <stdin>:10:17, :11:10, :12:5
  wire [7:0] _T_2 = 8'h40;      // <stdin>:13:15
  wire [63:0] _T_3 = TestEP_DataOut[_T_2+:64];  // <stdin>:14:10
  wire [5:0] _T_4 = 6'h20;      // <stdin>:15:16
  wire [31:0] _T_5 = /*cast(bit[31:0])*/_T_3[_T_4+:32]; // <stdin>:16:10, :17:10
  assert(_T_5 == 32'h0);        // <stdin>:18:15, :19:10, :20:5
  wire [5:0] _T_6 = 6'h10;      // <stdin>:21:15
  wire [15:0] _T_7 = /*cast(bit[15:0])*/_T_3[_T_6+:16]; // <stdin>:22:10, :23:10
  assert(_T_7 == 16'h40);       // <stdin>:24:16, :25:11, :26:5
  wire [5:0] _T_8 = 6'h0;       // <stdin>:27:14
  wire [15:0] _T_9 = /*cast(bit[15:0])*/_T_3[_T_8+:16]; // <stdin>:28:11, :29:11
  assert(_T_9 == 16'h0);        // <stdin>:30:15, :31:11, :32:5
  wire [7:0] _T_10 = 8'h30;     // <stdin>:33:15
endmodule
