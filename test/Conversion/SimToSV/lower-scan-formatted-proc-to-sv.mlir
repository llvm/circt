// RUN: circt-opt --lower-sim-to-sv %s | FileCheck %s

// CHECK-NOT: !sim.scan_string
// CHECK-NOT: sim.proc.scan
// CHECK-NOT: sim.scan.
// CHECK-NOT: sim.sv.channel_to_input_stream

// CHECK-LABEL: hw.module @scan_decimal
hw.module @scan_decimal(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %x = sv.reg : !hw.inout<i32>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.dec %x : !hw.inout<i32>
    // CHECK: sv.fscanf {{%.+}}, "%d"
    // CHECK-NOT: sim.proc.scan
    // CHECK-NOT: sim.scan.dec
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_binary
hw.module @scan_binary(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %x = sv.reg : !hw.inout<i32>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.bin %x : !hw.inout<i32>
    // CHECK: sv.fscanf {{%.+}}, "%b"
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_octal
hw.module @scan_octal(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %x = sv.reg : !hw.inout<i32>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.oct %x : !hw.inout<i32>
    // CHECK: sv.fscanf {{%.+}}, "%o"
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_hex_lower
hw.module @scan_hex_lower(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %x = sv.reg : !hw.inout<i32>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.hex %x : !hw.inout<i32> isUpper false
    // CHECK: sv.fscanf {{%.+}}, "%x"
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_hex_upper
hw.module @scan_hex_upper(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %x = sv.reg : !hw.inout<i32>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.hex %x : !hw.inout<i32> isUpper true
    // CHECK: sv.fscanf {{%.+}}, "%X"
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_real
hw.module @scan_real(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %r = sv.reg : !hw.inout<i64>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.real %r : !hw.inout<i64>
    // CHECK: sv.fscanf {{%.+}}, "%g"
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_time
hw.module @scan_time(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %r = sv.reg : !hw.inout<i64>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.time %r : !hw.inout<i64>
    // CHECK: sv.fscanf {{%.+}}, "%t"
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_char
hw.module @scan_char(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %c = sv.reg : !hw.inout<i8>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.char %c : !hw.inout<i8>
    // CHECK: sv.fscanf {{%.+}}, "%c"
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_unformatted_2val
hw.module @scan_unformatted_2val(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %x = sv.reg : !hw.inout<i32>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.unformatted %x : !hw.inout<i32>
    // CHECK: sv.fscanf {{%.+}}, "%u"
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_unformatted_4val
hw.module @scan_unformatted_4val(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %x = sv.reg : !hw.inout<i32>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.unformatted %x : !hw.inout<i32> four_value
    // CHECK: sv.fscanf {{%.+}}, "%z"
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_with_max_width
hw.module @scan_with_max_width(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %x = sv.reg : !hw.inout<i32>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.dec %x : !hw.inout<i32> width 8
    // CHECK: sv.fscanf {{%.+}}, "%8d"
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_suppressed
hw.module @scan_suppressed(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %x = sv.reg : !hw.inout<i32>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s_skip = sim.scan.dec {}
    %s_keep = sim.scan.dec %x : !hw.inout<i32>
    %fmt = sim.scan.concat (%s_skip, %s_keep)
    // CHECK: {{%.+}} = sv.fscanf {{%.+}}, "%*d%d"(%{{.+}}) : !hw.inout<i32>
    %count = sim.proc.scan %stream %fmt
  }
}

// CHECK-LABEL: hw.module @scan_literal_text
hw.module @scan_literal_text(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %x = sv.reg : !hw.inout<i32>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %lit = sim.scan.literal "val="
    %s = sim.scan.dec %x : !hw.inout<i32>
    %fmt = sim.scan.concat (%lit, %s)
    // CHECK: sv.fscanf {{%.+}}, "val=%d"
    %count = sim.proc.scan %stream %fmt
  }
}

// CHECK-LABEL: hw.module @scan_percent_in_literal
hw.module @scan_percent_in_literal(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %stream = sim.sv.channel_to_input_stream %fd_in
    %lit = sim.scan.literal "100%"
    // CHECK: {{%.+}} = sv.fscanf {{%.+}}, "100%%"
    %count = sim.proc.scan %stream %lit
  }
}

// CHECK-LABEL: hw.module @scan_hier_path
hw.module @scan_hier_path(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %x = sv.reg : !hw.inout<i32>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s_hier = sim.scan.hier_path_match
    %s_dec = sim.scan.dec %x : !hw.inout<i32>
    %fmt = sim.scan.concat (%s_hier, %s_dec)
    // CHECK: {{%.+}} = sv.fscanf {{%.+}}, "%m%d"(%{{.+}}) : !hw.inout<i32>
    %count = sim.proc.scan %stream %fmt
  }
}

// CHECK-LABEL: hw.module @scan_count_used
hw.module @scan_count_used(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %x = sv.reg : !hw.inout<i32>
    %done = sv.reg : !hw.inout<i1>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.dec %x : !hw.inout<i32>
    // CHECK: %[[RES:.+]] = sv.fscanf {{%.+}}, "%d"(%{{.+}}) : !hw.inout<i32>
    // CHECK-NEXT: %[[REG:.+]] = sv.reg
    // CHECK-NEXT: sv.bpassign %[[REG]], %[[RES]] : i32
    // CHECK-NEXT: %[[VAL:.+]] = sv.read_inout %[[REG]]
    %count = sim.proc.scan %stream %s
    %c1 = hw.constant 1 : i32
    %ok = comb.icmp eq %count, %c1 : i32
    sv.bpassign %done, %ok : i1
  }
}

// CHECK-LABEL: hw.module @scan_multiple_ops
hw.module @scan_multiple_ops(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %a = sv.reg : !hw.inout<i32>
    %b = sv.reg : !hw.inout<i32>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %sa = sim.scan.dec %a : !hw.inout<i32>
    %sb = sim.scan.hex %b : !hw.inout<i32> isUpper false
    // CHECK: sv.fscanf {{%.+}}, "%d"
    %count_a = sim.proc.scan %stream %sa
    // CHECK: sv.fscanf {{%.+}}, "%x"
    %count_b = sim.proc.scan %stream %sb
  }
}

// CHECK-LABEL: hw.module @all_scan_fragments
hw.module @all_scan_fragments(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %i32_reg = sv.reg : !hw.inout<i32>
    %i8_reg = sv.reg : !hw.inout<i8>
    %f64_reg = sv.reg : !hw.inout<i64>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s_lit = sim.scan.literal "prefix="
    %s_dec = sim.scan.dec %i32_reg : !hw.inout<i32>
    %s_bin = sim.scan.bin %i32_reg : !hw.inout<i32>
    %s_oct = sim.scan.oct %i32_reg : !hw.inout<i32>
    %s_hexl = sim.scan.hex %i32_reg : !hw.inout<i32> isUpper false
    %s_hexu = sim.scan.hex %i32_reg : !hw.inout<i32> isUpper true
    %s_real = sim.scan.real %f64_reg : !hw.inout<i64>
    %s_char = sim.scan.char %i8_reg : !hw.inout<i8>
    %s_hier = sim.scan.hier_path_match
    %fmt = sim.scan.concat (%s_lit, %s_dec, %s_bin, %s_oct, %s_hexl, %s_hexu, %s_real, %s_char, %s_hier)
    // CHECK: {{%.+}} = sv.fscanf {{%.+}}, "prefix=%d%b%o%x%X%g%c%m"(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : !hw.inout<i32>, !hw.inout<i32>, !hw.inout<i32>, !hw.inout<i32>, !hw.inout<i32>, !hw.inout<i64>, !hw.inout<i8>
    %count = sim.proc.scan %stream %fmt
  }
}

// CHECK-LABEL: hw.module @scan_str_basic
hw.module @scan_str_basic(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %buf = sv.reg : !hw.inout<i32>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.str %buf : !hw.inout<i32>
    // CHECK: sv.fscanf {{%.+}}, "%s"(%{{.+}}) : !hw.inout<i32>
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_str_suppressed
hw.module @scan_str_suppressed(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.str {}
    // CHECK: sv.fscanf {{%.+}}, "%*s"
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_str_width
hw.module @scan_str_width(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %buf = sv.reg : !hw.inout<i32>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.str %buf : !hw.inout<i32> width 16
    // CHECK: sv.fscanf {{%.+}}, "%16s"(%{{.+}}) : !hw.inout<i32>
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_unformatted_suppressed_2val
hw.module @scan_unformatted_suppressed_2val(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.unformatted {}
    // CHECK: sv.fscanf {{%.+}}, "%*u"
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_unformatted_suppressed_4val
hw.module @scan_unformatted_suppressed_4val(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.unformatted four_value
    // CHECK: sv.fscanf {{%.+}}, "%*z"
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_suppressed_with_width
hw.module @scan_suppressed_with_width(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %x = sv.reg : !hw.inout<i32>
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s_skip = sim.scan.dec width 8
    %s_keep = sim.scan.dec %x : !hw.inout<i32>
    %fmt = sim.scan.concat (%s_skip, %s_keep)
    // CHECK: sv.fscanf {{%.+}}, "%*8d%d"(%{{.+}}) : !hw.inout<i32>
    %count = sim.proc.scan %stream %fmt
  }
}

// CHECK-LABEL: hw.module @scan_literal_newline
hw.module @scan_literal_newline(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.literal "line\nend"
    // CHECK: sv.fscanf {{%.+}}, "line\0Aend"
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_literal_backslash
hw.module @scan_literal_backslash(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.literal "path\\to"
    // CHECK: sv.fscanf {{%.+}}, "path\\to"
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_hier_path_only
hw.module @scan_hier_path_only(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %stream = sim.sv.channel_to_input_stream %fd_in
    %s = sim.scan.hier_path_match
    // CHECK: sv.fscanf {{%.+}}, "%m"
    // CHECK-NOT: sv.fscanf {{%.+}}, "%m"(
    %count = sim.proc.scan %stream %s
  }
}

// CHECK-LABEL: hw.module @scan_empty_concat
hw.module @scan_empty_concat(in %clk : i1, in %fd : i32) {
  hw.triggered posedge %clk (%fd) : i32 {
    ^bb0(%fd_in : i32):
    %stream = sim.sv.channel_to_input_stream %fd_in
    %fmt = sim.scan.concat ()
    // CHECK: sv.fscanf {{%.+}}, ""
    %count = sim.proc.scan %stream %fmt
  }
}

// CHECK-LABEL: hw.module @scan_sscanf_basic
hw.module @scan_sscanf_basic(in %clk : i1) {
  hw.triggered posedge %clk {
    ^bb0:
    %str = sim.string.literal "42 0xAB"
    %x = sv.reg : !hw.inout<i32>
    %y = sv.reg : !hw.inout<i32>
    %stream = sim.string_to_input_stream %str
    %sd = sim.scan.dec %x : !hw.inout<i32>
    %sh = sim.scan.hex %y : !hw.inout<i32> isUpper false
    %fmt = sim.scan.concat (%sd, %sh)
    // CHECK: sv.sscanf "42 0xAB", "%d%x"({{.+}}, {{.+}}) : !hw.inout<i32>, !hw.inout<i32>
    %count = sim.proc.scan %stream %fmt
  }
}
