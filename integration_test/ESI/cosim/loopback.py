#!/usr/bin/python3

import binascii
import random
import cosim


class LoopbackTester(cosim.CosimBase):
  """Provides methods to test the loopback simulations."""

  def test_list(self):
    ifaces = self.cosim.list().wait().ifaces
    assert len(ifaces) > 0

  def test_open_close(self):
    ifaces = self.cosim.list().wait().ifaces
    openResp = self.cosim.open(ifaces[0]).wait()
    assert openResp.iface is not None
    ep = openResp.iface
    ep.close().wait()

  def test_i32(self, num_msgs):
    ep = self.openEP(sendType=self.schema.I32, recvType=self.schema.I32)
    for _ in range(num_msgs):
      data = random.randint(0, 2**32)
      print(f"Sending {data}")
      ep.send(self.schema.I32.new_message(i=data))
      result = self.readMsg(ep, self.schema.I32)
      print(f"Got {result}")
      assert (result.i == data)

  def write_3bytes(self, ep):
    r = random.randrange(0, 2**24)
    data = r.to_bytes(3, 'big')
    print(f'Sending: {binascii.hexlify(data)}')
    ep.send(self.schema.UntypedData.new_message(data=data)).wait()
    return data

  def read_3bytes(self, ep):
    dataMsg = self.readMsg(ep, self.schema.UntypedData)
    data = dataMsg.data
    print(binascii.hexlify(data))
    return data

  def test_3bytes(self, num_msgs=50):
    ep = self.openEP()
    print("Testing writes")
    dataSent = list()
    for _ in range(num_msgs):
      dataSent.append(self.write_3bytes(ep))
    print()
    print("Testing reads")
    dataRecv = list()
    for _ in range(num_msgs):
      dataRecv.append(self.read_3bytes(ep))
    ep.close().wait()
    assert dataSent == dataRecv
