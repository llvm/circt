module assertion_example (
    input logic clk,
    input logic rst_n,
    input logic req,
    input logic gnt,
    input logic [7:0] addr,
    input logic [31:0] data,
    input logic [1:0] burst_type,
    input logic burst_ready,
    input logic [3:0] burst_len,
    input logic error,
    input logic complete
);

    // Basic sequence: req must be followed by gnt within 1-5 cycles
    sequence req_to_gnt;
        req ##[1:5] gnt;
    endsequence

    // Burst transfer sequence
    // Note: Changed $repeat to explicit sequence for better tool compatibility
    sequence burst_transfer;
        burst_ready ##1 (data != 0) [*4];  // Using consecutive repetition operator
    endsequence

    // Address phase sequence
    sequence addr_phase;
        (addr != 0) ##[1:3] gnt;
    endsequence

    // Valid address range sequence
    sequence valid_addr_range;
        (addr inside {[8'h00:8'h7F]}) and (addr != 8'h10);
    endsequence

    // Error handling sequence
    sequence error_handle;
        error ##[1:3] complete;
    endsequence

    // Complex transfer sequence
    sequence complex_transfer;
        req ##1 (addr inside {[8'h00:8'h7F]}) ##[1:3] gnt ##1 
        burst_ready ##1 (data != 0) [*4] ##1 complete;
    endsequence

    // Property with local variable to check burst data progression
    property burst_data_check;
        logic [31:0] first_data;
        // (req && (burst_type == 2'b01), first_data = data) |=>
        (req && (burst_type == 2'b01)) |=>
        (burst_ready && (data > first_data));
    endproperty

    // Address consistency property
    property addr_consistency;
        logic [7:0] start_addr;
        // (req, start_addr = addr) |-> 
        req |-> 
        ##[1:5] (gnt && (addr == start_addr));
    endproperty

    // Reset condition property
    property no_req_during_reset;
        @(posedge clk) disable iff (!rst_n)
        !req;
    endproperty

    // Grant eventually property
    property gnt_eventually;
        @(posedge clk) req |-> ##[1:5] gnt;
    endproperty

    // Burst complete property
    property burst_complete;
        @(posedge clk) (req && (burst_len > 0)) |-> 
        ##[1:10] (burst_ready ##1 complete);
    endproperty

    // Simplified complex protocol property
    property complex_protocol;
        @(posedge clk) req |-> 
        ##[1:3] (gnt ##1 burst_ready ##1 complete);
    endproperty

    // Next cycle check property
    property next_cycle_check;
        @(posedge clk) req |-> ##1 (addr != 8'h00);
    endproperty

    // Modified strong complete property
    property complete_eventually;
        @(posedge clk) req |-> ##[1:20] complete;
    endproperty

    // moore.procedure always {
    //   %0 = moore.assertion_instance "req_to_gnt" {
    //     %3 = moore.read %req_2 : <l1>
    //     %4 = moore.read %gnt_3 : <l1>
    //     %5 = moore.int.ltl.delay %4, 1, 4 : (!moore.l1) -> !moore.i1
    //     %6 = moore.int.ltl.concat %3, %5 : (!moore.l1, !moore.i1) -> !moore.i1
    //     moore.lemma %6 : i1
    //   } : i1
    //   %1 = moore.read %clk_0 : <l1>
    //   %2 = moore.int.ltl.clock %0, posedge %1 : (!moore.i1, !moore.l1) -> !moore.i1
    //   moore.concurrrent_assert %2 : i1
    //   moore.return
    // }

    // Concurrent assertions
    ap_req_gnt: assert property (
        @(posedge clk) disable iff (!rst_n)
        req_to_gnt
    );

    // moore.procedure always {
    //   %0 = moore.assertion_instance "addr_consistency" {
    //     %start_addr = moore.variable : <l8>
    //     %3 = moore.read %req_2 : <l1>
    //     %4 = moore.read %gnt_3 : <l1>
    //     %5 = moore.read %addr_4 : <l8>
    //     %6 = moore.read %start_addr : <l8>
    //     %7 = moore.eq %5, %6 : l8 -> l1
    //     %8 = moore.and %4, %7 : l1
    //     %9 = moore.int.ltl.delay %8, 1, 4 : (!moore.l1) -> !moore.i1
    //     %10 = moore.int.ltl.implication %3, %9 : (!moore.l1, !moore.i1) -> !moore.i1
    //     moore.lemma %10 : i1
    //   } : i1
    //   %1 = moore.read %clk_0 : <l1>
    //   %2 = moore.int.ltl.clock %0, posedge %1 : (!moore.i1, !moore.l1) -> !moore.i1
    //   moore.concurrrent_assert %2 : i1
    //   moore.return
    // }

    ap_addr_consistent: assert property (
        @(posedge clk) disable iff (!rst_n)
        addr_consistency
    );

    // moore.procedure always {
    //   %0 = moore.read %req_2 : <l1>
    //   %1 = moore.read %gnt_3 : <l1>
    //   %2 = moore.int.ltl.delay %1, 1, 2 : (!moore.l1) -> !moore.i1
    //   %3 = moore.int.ltl.implication %0, %2 : (!moore.l1, !moore.i1) -> !moore.i1
    //   %4 = moore.read %clk_0 : <l1>
    //   %5 = moore.int.ltl.clock %3, posedge %4 : (!moore.i1, !moore.l1) -> !moore.i1
    //   moore.concurrrent_assert %5 : i1
    //   moore.return
    // }

    ap_basic_protocol: assert property (
        @(posedge clk) disable iff (!rst_n)
        req |-> ##[1:3] gnt
    );
    // moore.procedure always {
    //   %0 = moore.assertion_instance "burst_data_check" {
    //     %first_data = moore.variable : <l32>
    //     %3 = moore.read %req_2 : <l1>
    //     %4 = moore.read %burst_type_6 : <l2>
    //     %5 = moore.constant 1 : i2
    //     %6 = moore.conversion %5 : !moore.i2 -> !moore.l2
    //     %7 = moore.eq %4, %6 : l2 -> l1
    //     %8 = moore.and %3, %7 : l1
    //     %9 = moore.read %burst_ready_7 : <l1>
    //     %10 = moore.read %data_5 : <l32>
    //     %11 = moore.read %first_data : <l32>
    //     %12 = moore.ugt %10, %11 : l32 -> l1
    //     %13 = moore.and %9, %12 : l1
    //     %14 = moore.int.ltl.delay %13, 1, 0 : (!moore.l1) -> !moore.i1
    //     %15 = moore.int.ltl.implication %8, %14 : (!moore.l1, !moore.i1) -> !moore.i1
    //     moore.lemma %15 : i1
    //   } : i1
    //   %1 = moore.read %clk_0 : <l1>
    //   %2 = moore.int.ltl.clock %0, posedge %1 : (!moore.i1, !moore.l1) -> !moore.i1
    //   moore.concurrrent_assert %2 : i1
    //   moore.return
    // }

    ap_burst_data: assert property (
        @(posedge clk) disable iff (!rst_n)
        burst_data_check
    );

    // moore.procedure always {
    //   %0 = moore.read %error_9 : <l1>
    //   %1 = moore.read %complete_10 : <l1>
    //   %2 = moore.int.ltl.delay %1, 1, 4 : (!moore.l1) -> !moore.i1
    //   %3 = moore.int.ltl.implication %0, %2 : (!moore.l1, !moore.i1) -> !moore.i1
    //   %4 = moore.read %clk_0 : <l1>
    //   %5 = moore.int.ltl.clock %3, posedge %4 : (!moore.i1, !moore.l1) -> !moore.i1
    //   moore.concurrrent_assert %5 : i1
    //   moore.return
    // }

    ap_error: assert property (
        @(posedge clk) disable iff (!rst_n)
        error |-> ##[1:5] complete
    );
    // moore.procedure always {
    //   %0 = moore.read %req_2 : <l1>
    //   %1 = moore.read %burst_type_6 : <l2>
    //   %2 = moore.constant 0 : i2
    //   %3 = moore.conversion %2 : !moore.i2 -> !moore.l2
    //   %4 = moore.wildcard_eq %1, %3 : l2 -> l1
    //   %5 = moore.constant 1 : i2
    //   %6 = moore.conversion %5 : !moore.i2 -> !moore.l2
    //   %7 = moore.wildcard_eq %1, %6 : l2 -> l1
    //   %8 = moore.constant -2 : i2
    //   %9 = moore.conversion %8 : !moore.i2 -> !moore.l2
    //   %10 = moore.wildcard_eq %1, %9 : l2 -> l1
    //   %11 = moore.constant -1 : i2
    //   %12 = moore.conversion %11 : !moore.i2 -> !moore.l2
    //   %13 = moore.wildcard_eq %1, %12 : l2 -> l1
    //   %14 = moore.or %10, %13 : l1
    //   %15 = moore.or %7, %14 : l1
    //   %16 = moore.or %4, %15 : l1
    //   %17 = moore.and %0, %16 : l1
    //   %18 = moore.read %clk_0 : <l1>
    //   %19 = moore.int.ltl.clock %17, posedge %18 : (!moore.l1, !moore.l1) -> !moore.l1
    //   moore.concurrrent_assert %19 : l1
    //   moore.return
    // }

    // Coverage properties
    cp_burst_types: cover property (
        @(posedge clk) disable iff (!rst_n)
        (req && (burst_type inside {2'b00, 2'b01, 2'b10, 2'b11}))
    );
    // moore.procedure always {
    //   %0 = moore.read %burst_len_8 : <l4>
    //   %1 = moore.constant 1 : i4
    //   %2 = moore.conversion %1 : !moore.i4 -> !moore.l4
    //   %3 = moore.constant -8 : i4
    //   %4 = moore.conversion %3 : !moore.i4 -> !moore.l4
    //   %5 = moore.uge %0, %2 : l4 -> l1
    //   %6 = moore.ule %0, %4 : l4 -> l1
    //   %7 = moore.and %5, %6 : l1
    //   %8 = moore.read %clk_0 : <l1>
    //   %9 = moore.int.ltl.clock %7, posedge %8 : (!moore.l1, !moore.l1) -> !moore.l1
    //   moore.concurrrent_assert %9 : l1
    //   moore.return
    // }

    // Assumptions
    assume_valid_burst: assume property (
        @(posedge clk) disable iff (!rst_n)
        burst_len inside {[4'h1:4'h8]}
    );

    // moore.procedure always {
    //   %0 = moore.read %req_2 : <l1>
    //   %1 = moore.read %addr_4 : <l8>
    //   %2 = moore.constant 0 : i8
    //   %3 = moore.conversion %2 : !moore.i8 -> !moore.l8
    //   %4 = moore.constant 127 : i8
    //   %5 = moore.conversion %4 : !moore.i8 -> !moore.l8
    //   %6 = moore.uge %1, %3 : l8 -> l1
    //   %7 = moore.ule %1, %5 : l8 -> l1
    //   %8 = moore.and %6, %7 : l1
    //   %9 = moore.int.ltl.implication %0, %8 : (!moore.l1, !moore.l1) -> !moore.i1
    //   %10 = moore.read %clk_0 : <l1>
    //   %11 = moore.int.ltl.clock %9, posedge %10 : (!moore.i1, !moore.l1) -> !moore.i1
    //   moore.concurrrent_assert %11 : i1
    //   moore.return
    // }

    // Address range check
    ap_valid_addr: assert property (
        @(posedge clk) disable iff (!rst_n)
        req |-> (addr inside {[8'h00:8'h7F]})
    );

endmodule