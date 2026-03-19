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

// CHECK: moore.global_variable @"resource_db::db"
class resource_db
    #(type T);
  static T db[string];

  // CHECK-LABEL: func.func private @set(
  // CHECK: [[DB_REF:%.*]] = moore.get_global_variable @"resource_db::db"
  // CHECK: [[ENTRY_REF:%.*]] = moore.assoc_array_extract_ref [[DB_REF]][%arg0]
  // CHECK: moore.blocking_assign [[ENTRY_REF]], %arg1
  static function void set
      (string key,
       T value);
    db[key] = value;
  endfunction

  // CHECK-LABEL: func.func private @get(
  // CHECK: [[DB_REF2:%.*]] = moore.get_global_variable @"resource_db::db"
  // CHECK: [[EXISTS:%.*]] = moore.assoc_array.exists %arg0 in [[DB_REF2]]
  // CHECK: cf.cond_br
  // CHECK: [[DB_REF3:%.*]] = moore.get_global_variable @"resource_db::db"
  // CHECK: [[DB_VAL:%.*]] = moore.read [[DB_REF3]]
  // CHECK: [[VALUE:%.*]] = moore.assoc_array_extract [[DB_VAL]][%arg0]
  // CHECK: moore.blocking_assign %arg1, [[VALUE]]
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
