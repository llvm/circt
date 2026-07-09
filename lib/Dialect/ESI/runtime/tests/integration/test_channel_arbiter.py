#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Cosim integration tests for `esiaccel.components.ChannelArbiter`.

Runs under Verilator via the `@cosim_test` harness. Two DUTs (built by
`hw/channel_arbiter.py`) are exercised:

  * `arbiter_test` / `arbiter_test_odd`: host-driven single-flit multiplexers
    with a power-of-two ('balanced') and non-power-of-two ('unbalanced') input
    count. The host writes distinct tagged values to each input and checks that
    every value comes out of the single output exactly once (no loss /
    duplication) and that every input is served.

  * `list_test`: two in-hardware list-window producers contend for the
    arbiter; a hardware checker verifies message contiguity and streams a
    per-message report back to the host.

  * `token_test`: several in-hardware zero-width (`i0`) token producers, each
    emitting a fixed number of tokens, contend for the arbiter. A hardware
    counter reports a running ordinal per delivered token so the host can
    confirm every token is delivered exactly once and every producer is served.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import esiaccel
from esiaccel.accelerator import AcceleratorConnection
from esiaccel.cosim.pytest import cosim_test

HW_DIR = Path(__file__).resolve().parent / "hw"

NUM_INPUTS = 4  # power-of-two ("balanced") input count.
ODD_NUM_INPUTS = 3  # non-power-of-two ("unbalanced") input count.
PIPE_NUM_INPUTS = 6  # input count for the pipelined-mux-tree variant.
TOKEN_NUM_INPUTS = 5  # input count for the zero-width (i0) token variant.
TOKENS_PER_INPUT = 8  # tokens each producer emits in the token test.


def _check_mux(conn: AcceleratorConnection, dut_name: str,
               num_inputs: int) -> None:
  """Drive an N-input host mux and check every value is delivered exactly
  once and every input is served."""
  acc = conn.build_accelerator()
  dut = acc.children[esiaccel.AppID(dut_name)]
  ins = [dut.ports[esiaccel.AppID(f"in_{i}")] for i in range(num_inputs)]
  out = dut.ports[esiaccel.AppID("out")]
  for p in ins:
    p.connect()
  out.connect()

  # The value encodes its source input in bits [17:16] and a round counter in
  # the low bits, so the received multiset uniquely identifies every message.
  rounds = 6
  writes = [
      ((i << 16) | r, ins[i]) for r in range(rounds) for i in range(num_inputs)
  ]

  # Keep the number of in-flight (written-but-not-yet-read) messages strictly
  # below the output FIFO depth. This keeps a couple of inputs backlogged at
  # once (so the round-robin arbiter has to choose between them) while
  # avoiding the write-a-burst-before-reading deadlock.
  max_in_flight = 2
  sent: list[int] = []
  recv: list[int] = []
  wi = 0
  while len(recv) < len(writes):
    while wi < len(writes) and (len(sent) - len(recv)) < max_in_flight:
      value, port = writes[wi]
      port.write(value)
      sent.append(value)
      wi += 1
    recv.append(out.read().result())

  assert sorted(recv) == sorted(sent), \
      "arbiter dropped, duplicated or corrupted a value"

  # Every input (encoded in bits [17:16]) is served exactly `rounds` times.
  by_src = Counter(v >> 16 for v in recv)
  for i in range(num_inputs):
    assert by_src[i] == rounds, \
        f"input {i} served {by_src[i]} times, expected {rounds}"


@cosim_test(HW_DIR / "channel_arbiter.py")
class TestChannelArbiterCosim:

  def test_mux_correctness(self, conn: AcceleratorConnection) -> None:
    """Balanced (power-of-two) input count: every value appears once."""
    _check_mux(conn, "arbiter_test", NUM_INPUTS)

  def test_mux_correctness_unbalanced(self,
                                      conn: AcceleratorConnection) -> None:
    """Unbalanced (non-power-of-two) input count: the array-indexed mux over
    N < 2**clog2(N) elements and the round-robin wrap still deliver every
    value exactly once."""
    _check_mux(conn, "arbiter_test_odd", ODD_NUM_INPUTS)

  def test_mux_correctness_pipelined(self, conn: AcceleratorConnection) -> None:
    """Pipelined selection mux tree: the multi-cycle mux latency (absorbed by a
    deeper output FIFO + credit counter) must still deliver every value exactly
    once."""
    _check_mux(conn, "arbiter_test_pipe", PIPE_NUM_INPUTS)

  def test_list_contiguity(self, conn: AcceleratorConnection) -> None:
    """Contending multi-flit list messages are never interleaved."""
    acc = conn.build_accelerator()
    dut = acc.children[esiaccel.AppID("list_test")]
    report = dut.ports[esiaccel.AppID("report")]
    report.connect()

    seen_src: set[int] = set()
    num_reports = 40
    for _ in range(num_reports):
      value = report.read().result()
      err = (value >> 24) & 0x1
      src = (value >> 16) & 0xff
      assert err == 0, \
          f"hardware detected interleaved list flits (report {value:#010x})"
      seen_src.add(src)

    # Both contending producers (src 1 and src 2) must get through.
    assert seen_src == {1, 2}, \
        f"expected both sources to be served, saw {sorted(seen_src)}"

  def test_token_conservation(self, conn: AcceleratorConnection) -> None:
    """Zero-width (`i0`) token payloads: the credit-counter-as-buffer path
    delivers every token exactly once (no loss/duplication/reorder) and serves
    every producer -- the total is only reachable if no producer is starved."""
    acc = conn.build_accelerator()
    dut = acc.children[esiaccel.AppID("token_test")]
    report = dut.ports[esiaccel.AppID("token_report")]
    report.connect()

    total = TOKEN_NUM_INPUTS * TOKENS_PER_INPUT
    for expected in range(1, total + 1):
      value = report.read().result()
      assert value == expected, \
          f"token {expected} arrived as {value} (loss / duplication / reorder)"
