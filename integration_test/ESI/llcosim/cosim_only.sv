// REQUIRES: esi-llcosim
// RUN: esi-cosim-runner.py --low-level %s %s
// PY: import llcosim_test
// PY: llcosim_test.test(rpcschemapath, simhostport)

import LLCosim_DpiPkg::*;

module top (
    input logic clk,
    input logic rst
);

  initial
  begin
    int initrc = llcosim_init();
    assert (initrc == 0);
  end
endmodule
