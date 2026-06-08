// RUN: circt-opt -export-verilog %s | FileCheck %s

// CHECK-LABEL: module fscanf_no_count(
// CHECK: reg [31:0] x;
// CHECK: always @(posedge clk)
// CHECK-NEXT: $fscanf(fd, "%d", x);
hw.module @fscanf_no_count(in %clk : i1, in %fd : i32) {
  %x = sv.reg : !hw.inout<i32>
  sv.always posedge %clk {
    %0 = sv.fscanf %fd, "%d"(%x) : !hw.inout<i32>
  }
  hw.output
}

// CHECK-LABEL: module fscanf_with_count(
// CHECK: reg [31:0] x;
// CHECK: reg [31:0] count;
// CHECK: always @(posedge clk)
// CHECK-NEXT: count = $fscanf(fd, "%d", x);
hw.module @fscanf_with_count(in %clk : i1, in %fd : i32) {
  %x = sv.reg : !hw.inout<i32>
  %count = sv.reg : !hw.inout<i32>
  sv.always posedge %clk {
    %0 = sv.fscanf %fd, "%d"(%x) : !hw.inout<i32>
    sv.bpassign %count, %0 : i32
  }
  hw.output
}

// CHECK-LABEL: module fscanf_multiple_outputs(
// CHECK: $fscanf(fd, "%d%x", a, b);
hw.module @fscanf_multiple_outputs(in %clk : i1, in %fd : i32) {
  %a = sv.reg : !hw.inout<i32>
  %b = sv.reg : !hw.inout<i32>
  sv.always posedge %clk {
    %0 = sv.fscanf %fd, "%d%x"(%a, %b) : !hw.inout<i32>, !hw.inout<i32>
  }
  hw.output
}

// CHECK-LABEL: module fscanf_no_outputs(
// CHECK: $fscanf({{.+}}, "%m");
hw.module @fscanf_no_outputs(in %clk : i1, in %fd : i32) {
  sv.always posedge %clk {
    %0 = sv.fscanf %fd, "%m"
  }
  hw.output
}

// CHECK-LABEL: module fscanf_suppressed(
// CHECK: $fscanf({{.+}}, "%*d%d", {{.+}});
hw.module @fscanf_suppressed(in %clk : i1, in %fd : i32) {
  %x = sv.reg : !hw.inout<i32>
  sv.always posedge %clk {
    %0 = sv.fscanf %fd, "%*d%d"(%x) : !hw.inout<i32>
  }
  hw.output
}

// CHECK-LABEL: module fscanf_width(
// CHECK: $fscanf({{.+}}, "%8d", {{.+}});
hw.module @fscanf_width(in %clk : i1, in %fd : i32) {
  %x = sv.reg : !hw.inout<i32>
  sv.always posedge %clk {
    %0 = sv.fscanf %fd, "%8d"(%x) : !hw.inout<i32>
  }
  hw.output
}

// CHECK-LABEL: module fscanf_literal_text(
// CHECK: $fscanf({{.+}}, "val=%d", {{.+}});
hw.module @fscanf_literal_text(in %clk : i1, in %fd : i32) {
  %x = sv.reg : !hw.inout<i32>
  sv.always posedge %clk {
    %0 = sv.fscanf %fd, "val=%d"(%x) : !hw.inout<i32>
  }
  hw.output
}

// CHECK-LABEL: module fscanf_percent_literal(
// CHECK: $fscanf({{.+}}, "100%%");
hw.module @fscanf_percent_literal(in %clk : i1, in %fd : i32) {
  sv.always posedge %clk {
    %0 = sv.fscanf %fd, "100%%"
  }
  hw.output
}

// CHECK-LABEL: module fscanf_all_integer_specifiers(
// CHECK: $fscanf({{.+}}, "%d%b%o%x%X", {{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}});
hw.module @fscanf_all_integer_specifiers(in %clk : i1, in %fd : i32) {
  %dec = sv.reg : !hw.inout<i32>
  %bin = sv.reg : !hw.inout<i32>
  %oct = sv.reg : !hw.inout<i32>
  %hex_l = sv.reg : !hw.inout<i32>
  %hex_u = sv.reg : !hw.inout<i32>
  sv.always posedge %clk {
    %0 = sv.fscanf %fd, "%d%b%o%x%X"(%dec, %bin, %oct, %hex_l, %hex_u) : !hw.inout<i32>, !hw.inout<i32>, !hw.inout<i32>, !hw.inout<i32>, !hw.inout<i32>
  }
  hw.output
}

// CHECK-LABEL: module fscanf_count_in_always(
hw.module @fscanf_count_in_always(in %clk : i1, in %fd : i32) {
  %x = sv.reg : !hw.inout<i32>
  sv.always posedge %clk {
    %count = sv.reg : !hw.inout<i32>
    %0 = sv.fscanf %fd, "%d"(%x) : !hw.inout<i32>
    sv.bpassign %count, %0 : i32
  }
  hw.output
}

// CHECK-LABEL: module sscanf_basic(
hw.module @sscanf_basic(in %clk : i1) {
  %x = sv.reg : !hw.inout<i32>
  // CHECK: always @(posedge clk)
  sv.always posedge %clk {
    // CHECK-NEXT: $sscanf({{.+}}, "%d", x);
    %0 = sv.sscanf "hello", "%d"(%x) : !hw.inout<i32>
  }
  hw.output
}
