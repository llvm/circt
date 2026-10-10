//===- EsiDramModel.sv - Simple DRAM model for ESI cosimulation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A simple, behavioral DRAM model which implements the generic (Avalon-MM
// friendly) upstream interface of the ESI DRAM service ('ChannelDram' in
// esiaccel/bsp/dram.py). It is intended for cosimulation only and is written
// to be compatible with Verilator (as well as commercial simulators).
//
// Interface (all valid/ready; a transfer happens when both are high):
//   - Read request:   {address (word), burstcount, tag}. Returns 'burstcount'
//                     consecutive words starting at 'address'.
//   - Read response:  {tag, data, last}, one message per word. 'last' marks the
//                     final word of the burst.
//   - Write request:  {address (word), burstcount, tag, data, byteenable,
//                     last}, one message per word. 'address', 'burstcount' and
//                     'tag' are sampled on the first word of a burst, as in
//                     Avalon-MM. Only bytes with their 'byteenable' bit set
//                     are written.
//   - Write ack:      {tag}, one message per write burst.
//
// Timing model: every accepted request is scheduled to complete after a fixed
// latency plus a pseudo-random jitter in [0, JITTER] cycles. Completion times
// are clamped to be monotonic so responses are always returned in request order
// (per direction), as a real Avalon-MM memory controller would. Requests are
// back-pressured when the respective queue is full, and (optionally) randomly
// to exercise the handshake.
//
// Storage: a sparse associative array indexed by word address, so the model
// supports large address spaces cheaply. Unwritten words read as zero. Contents
// are *not* cleared by reset (like real DRAM). Writes take effect when
// accepted, so a read issued after a write's ack always observes the write.
//
//===----------------------------------------------------------------------===//

module EsiDramModel #(
    parameter int DATA_WIDTH = 64,
    parameter int ADDR_WIDTH = 32,
    parameter int BURST_WIDTH = 7,
    parameter int READ_LATENCY = 20,
    parameter int WRITE_LATENCY = 10,
    parameter int JITTER = 8,
    parameter int QUEUE_DEPTH = 16,
    // Percentage (0-100) of cycles on which request 'ready' is randomly
    // deasserted even when there is space in the queue.
    parameter int STALL_PERCENT = 0,
    parameter int SEED = 1
) (
    input logic clk,
    input logic rst,

    // Read request.
    input  logic                   rd_req_valid,
    output logic                   rd_req_ready,
    input  logic [ ADDR_WIDTH-1:0] rd_req_address,
    input  logic [BURST_WIDTH-1:0] rd_req_burstcount,
    input  logic [            7:0] rd_req_tag,

    // Read response.
    output logic                  rd_resp_valid,
    input  logic                  rd_resp_ready,
    output logic [           7:0] rd_resp_tag,
    output logic [DATA_WIDTH-1:0] rd_resp_data,
    output logic                  rd_resp_last,

    // Write request (one message per word).
    input  logic                    wr_req_valid,
    output logic                    wr_req_ready,
    input  logic [  ADDR_WIDTH-1:0] wr_req_address,
    input  logic [ BURST_WIDTH-1:0] wr_req_burstcount,
    input  logic [             7:0] wr_req_tag,
    input  logic [  DATA_WIDTH-1:0] wr_req_data,
    input  logic [DATA_WIDTH/8-1:0] wr_req_byteenable,
    input  logic                    wr_req_last,

    // Write ack (one message per burst).
    output logic       wr_ack_valid,
    input  logic       wr_ack_ready,
    output logic [7:0] wr_ack_tag
);

  localparam int BE_WIDTH = DATA_WIDTH / 8;
  localparam int QPTR_WIDTH = QUEUE_DEPTH > 1 ? $clog2(QUEUE_DEPTH) : 1;
  localparam int QCNT_WIDTH = $clog2(QUEUE_DEPTH + 1);

  // Sparse backing store, indexed by word address.
  logic [DATA_WIDTH-1:0] mem[bit [ADDR_WIDTH-1:0]];

  // Free-running cycle counter used for scheduling.
  longint unsigned now;

  // xorshift32 PRNG: deterministic and simulator-independent.
  logic [31:0] rng;
  function automatic logic [31:0] xorshift32(input logic [31:0] x);
    logic [31:0] y;
    y = x ^ (x << 13);
    y = y ^ (y >> 17);
    y = y ^ (y << 5);
    return y;
  endfunction

  function automatic longint unsigned jitter(input logic [31:0] r);
    if (JITTER <= 0) return 0;
    return longint'(r) % (longint'(JITTER) + 1);
  endfunction

  function automatic logic stall(input logic [31:0] r);
    if (STALL_PERCENT <= 0) return 1'b0;
    return (r % 32'd100) < 32'(STALL_PERCENT);
  endfunction

  function automatic logic [DATA_WIDTH-1:0] read_word(
      input logic [ADDR_WIDTH-1:0] addr);
    if (mem.exists(addr)) return mem[addr];
    return '0;
  endfunction

  function automatic logic [BURST_WIDTH-1:0] burst_len(
      input logic [BURST_WIDTH-1:0] burstcount);
    // A burstcount of 0 is illegal in Avalon-MM; treat it as a single word.
    return burstcount == 0 ? 1 : burstcount;
  endfunction

  //===--------------------------------------------------------------------===//
  // Read path.
  //===--------------------------------------------------------------------===//

  logic [ ADDR_WIDTH-1:0] rq_addr [QUEUE_DEPTH];
  logic [BURST_WIDTH-1:0] rq_len  [QUEUE_DEPTH];
  logic [            7:0] rq_tag  [QUEUE_DEPTH];
  longint unsigned        rq_time [QUEUE_DEPTH];
  logic [ QPTR_WIDTH-1:0] rq_head;
  logic [ QPTR_WIDTH-1:0] rq_tail;
  logic [ QCNT_WIDTH-1:0] rq_count;
  logic [BURST_WIDTH-1:0] rd_beat;
  longint unsigned        rd_last_time;
  logic                   rd_stall;

  assign rd_req_ready = (32'(rq_count) < QUEUE_DEPTH) && !rd_stall;
  wire rd_req_xact = rd_req_valid && rd_req_ready;
  wire rd_resp_xact = rd_resp_valid && rd_resp_ready;
  // Load the next beat into the output register when it is empty or draining.
  wire rd_head_ready = rq_count != 0 && now >= rq_time[rq_head];
  wire rd_load = rd_head_ready && (!rd_resp_valid || rd_resp_ready);

  //===--------------------------------------------------------------------===//
  // Write path.
  //===--------------------------------------------------------------------===//

  logic [           7:0] wq_tag  [QUEUE_DEPTH];
  longint unsigned       wq_time [QUEUE_DEPTH];
  logic [QPTR_WIDTH-1:0] wq_head;
  logic [QPTR_WIDTH-1:0] wq_tail;
  logic [QCNT_WIDTH-1:0] wq_count;
  longint unsigned       wr_last_time;
  logic                  wr_stall;

  // In-progress write burst state.
  logic                   wr_in_burst;
  logic [ ADDR_WIDTH-1:0] wr_addr;
  logic [BURST_WIDTH-1:0] wr_remaining;
  logic [            7:0] wr_tag;

  assign wr_req_ready = (32'(wq_count) < QUEUE_DEPTH) && !wr_stall;
  wire wr_req_xact = wr_req_valid && wr_req_ready;
  wire wr_ack_xact = wr_ack_valid && wr_ack_ready;
  wire wr_ack_load = wq_count != 0 && now >= wq_time[wq_head] &&
      (!wr_ack_valid || wr_ack_ready);

  // Address, remaining count and tag of the current write beat.
  wire [ADDR_WIDTH-1:0] wr_beat_addr = wr_in_burst ? wr_addr : wr_req_address;
  wire [BURST_WIDTH-1:0] wr_beat_remaining =
      wr_in_burst ? wr_remaining : burst_len(wr_req_burstcount);
  wire [7:0] wr_beat_tag = wr_in_burst ? wr_tag : wr_req_tag;
  wire wr_beat_final = wr_beat_remaining == 1;

  always_ff @(posedge clk) begin
    if (rst) begin
      now <= 0;
      rng <= (SEED == 0) ? 32'h1 : 32'(SEED);
      rd_stall <= 1'b0;
      wr_stall <= 1'b0;

      rq_head <= '0;
      rq_tail <= '0;
      rq_count <= '0;
      rd_beat <= '0;
      rd_last_time <= 0;
      rd_resp_valid <= 1'b0;
      rd_resp_tag <= '0;
      rd_resp_data <= '0;
      rd_resp_last <= 1'b0;

      wq_head <= '0;
      wq_tail <= '0;
      wq_count <= '0;
      wr_last_time <= 0;
      wr_in_burst <= 1'b0;
      wr_addr <= '0;
      wr_remaining <= '0;
      wr_tag <= '0;
      wr_ack_valid <= 1'b0;
      wr_ack_tag <= '0;
    end else begin
      automatic logic [QCNT_WIDTH-1:0] rq_count_next = rq_count;
      automatic logic [QCNT_WIDTH-1:0] wq_count_next = wq_count;
      automatic logic [31:0] rng_next = xorshift32(rng);

      now <= now + 1;
      rng <= rng_next;
      rd_stall <= stall(rng_next);
      wr_stall <= stall(xorshift32(rng_next));

      //===--- Read request intake ---===//
      if (rd_req_xact) begin
        automatic longint unsigned t = now + longint'(READ_LATENCY) + jitter(rng);
        if (t < rd_last_time) t = rd_last_time;
        rq_addr[rq_tail] <= rd_req_address;
        rq_len[rq_tail] <= burst_len(rd_req_burstcount);
        rq_tag[rq_tail] <= rd_req_tag;
        rq_time[rq_tail] <= t;
        rd_last_time <= t;
        rq_tail <= (rq_tail == QPTR_WIDTH'(QUEUE_DEPTH - 1)) ? '0 : rq_tail + 1;
        rq_count_next = rq_count_next + 1;
`ifdef ESI_DRAM_DEBUG
        $display("[%0t] %m: read req addr=%h len=%0d tag=%0d", $time,
                 rd_req_address, burst_len(rd_req_burstcount), rd_req_tag);
`endif
      end

      //===--- Read response ---===//
      if (rd_resp_xact && !rd_load) rd_resp_valid <= 1'b0;
      if (rd_load) begin
        automatic logic last_beat = (rd_beat + 1) == rq_len[rq_head];
        rd_resp_valid <= 1'b1;
        rd_resp_tag <= rq_tag[rq_head];
        rd_resp_data <= read_word(rq_addr[rq_head] + ADDR_WIDTH'(rd_beat));
        rd_resp_last <= last_beat;
        if (last_beat) begin
          rd_beat <= '0;
          rq_head <= (rq_head == QPTR_WIDTH'(QUEUE_DEPTH - 1)) ? '0 : rq_head + 1;
          rq_count_next = rq_count_next - 1;
        end else begin
          rd_beat <= rd_beat + 1;
        end
      end
      rq_count <= rq_count_next;

      //===--- Write data intake ---===//
      if (wr_req_xact) begin
        automatic logic [DATA_WIDTH-1:0] word = read_word(wr_beat_addr);
        for (int i = 0; i < BE_WIDTH; i++)
          if (wr_req_byteenable[i]) word[i*8+:8] = wr_req_data[i*8+:8];
        // Dynamically-sized arrays cannot be assigned non-blocking. The read
        // path above has already sampled 'mem' for this cycle.
        /* verilator lint_off BLKSEQ */
        mem[wr_beat_addr] = word;
        /* verilator lint_on BLKSEQ */
`ifdef ESI_DRAM_DEBUG
        $display("[%0t] %m: write addr=%h data=%h be=%h tag=%0d", $time,
                 wr_beat_addr, wr_req_data, wr_req_byteenable, wr_beat_tag);
`endif
        if (wr_req_last != wr_beat_final)
          $error("%m: write 'last' (%0d) inconsistent with burstcount",
                 wr_req_last);

        if (wr_beat_final) begin
          automatic longint unsigned t =
              now + longint'(WRITE_LATENCY) + jitter(xorshift32(rng));
          if (t < wr_last_time) t = wr_last_time;
          wq_tag[wq_tail] <= wr_beat_tag;
          wq_time[wq_tail] <= t;
          wr_last_time <= t;
          wq_tail <= (wq_tail == QPTR_WIDTH'(QUEUE_DEPTH - 1)) ? '0 : wq_tail + 1;
          wq_count_next = wq_count_next + 1;
          wr_in_burst <= 1'b0;
        end else begin
          wr_in_burst <= 1'b1;
          wr_addr <= wr_beat_addr + 1;
          wr_remaining <= wr_beat_remaining - 1;
          wr_tag <= wr_beat_tag;
        end
      end

      //===--- Write ack ---===//
      if (wr_ack_xact && !wr_ack_load) wr_ack_valid <= 1'b0;
      if (wr_ack_load) begin
        wr_ack_valid <= 1'b1;
        wr_ack_tag <= wq_tag[wq_head];
        wq_head <= (wq_head == QPTR_WIDTH'(QUEUE_DEPTH - 1)) ? '0 : wq_head + 1;
        wq_count_next = wq_count_next - 1;
      end
      wq_count <= wq_count_next;
    end
  end

endmodule
