#!/usr/bin/python3

import binascii
import random
import esi_cosim


class LoopbackTester(esi_cosim.CosimBase):
  """Provides methods to test the loopback simulations."""

  def test_list(self):
    ifaces = self.cosim.list().wait().ifaces
    assert len(ifaces) > 0

  def test_two_chan_loopback(self, num_msgs):
    to_hw = self.openEP("top.loopback_inst5B05D_loopback_tohw_recv")
    from_hw = self.openEP("top.loopback_inst5B05D_loopback_fromhw_send")
    for _ in range(num_msgs):
      i = random.randint(0, 2**8 - 1)
      print(f"Sending {i}")
      to_hw.sendFromHost(msg=i.to_bytes(1, 'little')).wait()
      result = self.readMsg(from_hw)
      result_int = int.from_bytes(result, 'little')
      print(f"Got {result_int}")
      assert (result_int == i)

  def write_3bytes(self, ep):
    r = random.randrange(0, 2**24 - 1)
    data = r.to_bytes(3, 'little')
    print(f'Sending: {r:8x} as {binascii.hexlify(data)}')
    ep.sendFromHost(msg=data).wait()
    return r

  def read_3bytes(self, ep):
    data = self.readMsg(ep)
    i = int.from_bytes(data, 'little')
    print(f"Recv'd: {i:8x} as {binascii.hexlify(data)}")
    return i

  def test_3bytes(self, num_msgs=50):
    send_ep = self.openEP("top.fromHost", from_host_type="i24")
    recv_ep = self.openEP("top.toHost", to_host_type="i32")
    print("Testing writes")
    dataSent = list()
    for _ in range(num_msgs):
      dataSent.append(self.write_3bytes(send_ep))
    print()
    print("Testing reads")
    dataRecv = list()
    for _ in range(num_msgs):
      dataRecv.append(self.read_3bytes(recv_ep))
    send_ep.close().wait()
    recv_ep.close().wait()
    assert dataSent == dataRecv


if __name__ == "__main__":
  import sys
  rpc = LoopbackTester(sys.argv[2], sys.argv[1])
  print(rpc.list())
  rpc.test_two_chan_loopback(25)
