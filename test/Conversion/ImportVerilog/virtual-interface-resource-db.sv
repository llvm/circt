// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang
//
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

interface output_if
    (input logic clk);
  logic [7 : 0] data;
endinterface

class resource_db
    #(type T);
  static T db[string];

  static function void set
      (string key,
       T value);
    db[key] = value;
  endfunction

  static function bit get
      (string key,
       output T value);
    if (db.exists(key)) begin
      value = db[key];
      return 1;
    end
    return 0;
  endfunction
endclass

class consumer;
  virtual output_if vif;

  // CHECK: moore.assoc_array.exists
  // CHECK: moore.assoc_array_extract
  function bit connect
      ();
    return resource_db #(virtual output_if)::get("out", vif);
  endfunction

  function logic [7 : 0]
    sample();
    return vif.data;
  endfunction

  function void drive_byte
      (logic [7 : 0] v);
    vif.data = v;
  endfunction
endclass

// CHECK-LABEL: moore.module @top
module top;
  logic clk;
  output_if out(clk);

  consumer c;
  logic [7 : 0] s;

  initial begin
    c = new;
    resource_db #(virtual output_if)::set("out", out);
    assert (c.connect());
    c.drive_byte(8'h3c);
    s = c.sample();
  end
endmodule
