; RUN: firtool %s | FileCheck %s

FIRRTL version 5.1.0
; CHECK: `ifndef SYNTHESIS
; CHECK: `ifndef __CIRCT_LIB_LOGGING
; CHECK-NEXT:   // CIRCT Logging Library
; CHECK-NEXT:   package __circt_lib_logging;
; CHECK-NEXT:     class FileDescriptor;
; CHECK-NEXT:       static int global_id [string];
; CHECK-NEXT:       static function int get(string name);
; CHECK-NEXT:         if (global_id.exists(name) == 32'h0) begin
; CHECK-NEXT:           global_id[name] = $fopen(name, "w");
; CHECK-NEXT:           if (global_id[name] == 32'h0)
; CHECK-NEXT:             $error("Failed to open file %s", name);
; CHECK-NEXT:         end
; CHECK-NEXT:         return global_id[name];
; CHECK-NEXT:       endfunction
; CHECK-NEXT:     endclass
; CHECK-NEXT:   endpackage
; CHECK-NEXT:   `define __CIRCT_LIB_LOGGING
; CHECK-NEXT: `endif // not def __CIRCT_LIB_LOGGING
; CHECK-NEXT: `endif // not def SYNTHESIS
circuit PrintTest:
  ; CHECK-LABEL: module PrintTest
  public module PrintTest :
    input clock : Clock
    input cond : UInt<1>
    input a : UInt<8>

    ; CHECK:      always @(posedge clock) begin
    ; CHECK-NEXT:   automatic logic [63:0] [[TIME_VAR:_.+]];
    ; CHECK-NEXT:   [[TIME_VAR]] = $time;
    ; CHECK-NEXT:   [[FD_TEST:_.+]] = __circt_lib_logging::FileDescriptor::get("test.txt");    
    ; CHECK-NEXT:   [[FD_NAME:_.+]] = __circt_lib_logging::FileDescriptor::get($sformatf("%m%c.txt",
    ; CHECK-NEXT:   8'h61));
    ; CHECK-NEXT:   [[FD_DYN:_.+]]  = __circt_lib_logging::FileDescriptor::get($sformatf("%0t%d.txt",
    ; CHECK-NEXT:   [[TIME_VAR]],
    ; CHECK-NEXT:   a));

    ; CHECK:        if (cond) begin
    ; CHECK:          `ifndef SYNTHESIS
    ; CHECK-NEXT:     if (`PRINTF_COND_) begin
    ; CHECK:            $fwrite(32'h80000002, "binary: %b %0b %8b\n", a, a, a);
    printf(clock, cond, "binary: %b %0b %8b\n", a, a, a)

    ; CHECK-NEXT:       $fwrite(32'h80000002, "decimal: %d %0d %3d\n", a, a, a);
    printf(clock, cond, "decimal: %d %0d %3d\n", a, a, a)

    ; CHECK-NEXT:       $fwrite(32'h80000002, "hexadecimal: %x %0x %2x\n", a, a, a);
    printf(clock, cond, "hexadecimal: %x %0x %2x\n", a, a, a)

    ; CHECK-NEXT:       $fwrite(32'h80000002, "ASCII character: %c\n", a);
    printf(clock, cond, "ASCII character: %c\n", a)

    ; CHECK-NEXT:       $fwrite(32'h80000002, "literals: %%\n");
    printf(clock, cond, "literals: %%\n")

    ; CHECK-NEXT:       $fwrite(32'h80000002, "[%0t]: %m\n", [[TIME_VAR]]);
    ; CHECK-NEXT:     end
    printf(clock, cond, "[{{SimulationTime}}]: {{HierarchicalModuleName}}\n")

    ; CHECK-NEXT:     $fwrite([[FD_TEST]], "hello");
    fprintf(clock, cond, "test.txt", "hello")

    ; CHECK-NEXT:     $fwrite([[FD_NAME]],
    ; CHECK-NEXT:         "[%0t]: static file name (w/ substitution)\n", [[TIME_VAR]]);
    node c = UInt<8>(97)
    fprintf(clock, cond, "{{HierarchicalModuleName}}%c.txt", c, "[{{SimulationTime}}]: static file name (w/ substitution)\n")

    ; CHECK-NEXT:     $fflush(32'h80000002);
    
    fflush(clock, cond)

    ; CHECK-NEXT:     $fflush([[FD_DYN]]);
    ; CHECK-NEXT:     `endif // not def SYNTHESIS
    ; CHECK-NEXT:   end
    fflush(clock, cond, "{{SimulationTime}}%d.txt", a);
