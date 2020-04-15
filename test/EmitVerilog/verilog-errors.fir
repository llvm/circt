; RUN: spt-translate -parse-fir %s --mlir-print-debuginfo | spt-translate -emit-verilog  -verify-diagnostics  


circuit Xorr :
  module Xorr :
    input a: UInt  ; expected-error {{xxx}}
 
