#!/usr/bin/python3

import esi_cosim
import random


class LoopbackTester(esi_cosim.CosimBase):
  """Provides methods to test the loopback simulations."""

  def test_two_chan_loopback(self, num_msgs):
    to_hw = self.openEP("TOP.top.Mid.Producer_loopback_in",
                        sendType=self.schema.I1,
                        recvType=self.schema.I32)
    from_hw = self.openEP("TOP.top.Mid.Consumer_loopback_out",
                          sendType=self.schema.I32,
                          recvType=self.schema.I1)
    for _ in range(num_msgs):
      data = random.randint(0, 2**32 - 1)
      print(f"Sending {data}")
      to_hw.send(self.schema.I32.new_message(i=data))
      result = self.readMsg(from_hw, self.schema.I32)
      print(f"Got {result}")
      assert (result.i == data)

  def test_one_chan_loopback(self, num_msgs):
    hw = self.openEP("TOP.top.Mid.LoopbackInOut_loopback_inout",
                     sendType=self.schema.I16,
                     recvType=self.schema.I32)
    for _ in range(num_msgs):
      data = random.randint(0, 2**32 - 1)
      print(f"Sending {data}")
      hw.send(self.schema.I32.new_message(i=data))
      result = self.readMsg(hw, self.schema.I16)
      print(f"Got {result}")
      assert (result.i == data & 0xFFFF)
