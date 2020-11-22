#!/usr/bin/python3

import binascii
import capnp
import random
import time


class LoopbackTester:
    """Provides methods to test the untyped loopback simulation."""

    # Some of these are generalizable and should be hoisted into a base class
    # when other cosim tests get written.

    def __init__(self, schemaPath, hostPort):
        self.dpi = capnp.load(schemaPath)
        self.rpc_client = capnp.TwoPartyClient(hostPort)
        self.cosim = self.rpc_client.bootstrap().cast_as(self.dpi.CosimDpiServer)

    def test_list(self):
        ifaces = self.cosim.list().wait().ifaces
        assert len(ifaces) > 0

    def test_open_close(self):
        ifaces = self.cosim.list().wait().ifaces
        openResp = self.cosim.open(ifaces[0]).wait()
        assert openResp.iface is not None
        ep = openResp.iface
        ep.close().wait()

    def openEP(self):
        ifaces = self.cosim.list().wait().ifaces
        openResp = self.cosim.open(ifaces[0]).wait()
        assert openResp.iface is not None
        return openResp.iface

    def write(self, ep):
        r = random.randrange(0, 2**24)
        data = r.to_bytes(3, 'big')
        print(f'Sending: {binascii.hexlify(data)}')
        ep.send(self.dpi.UntypedData.new_message(data=data)).wait()
        return data

    def read(self, ep):
        while True:
            recvResp = ep.recv(False).wait()
            if recvResp.hasData:
                break
            else:
                time.sleep(0.1)
        assert recvResp.resp is not None
        dataMsg = recvResp.resp.as_struct(self.dpi.UntypedData)
        data = dataMsg.data
        print(binascii.hexlify(data))
        return data

    def write_read(self):
        ep = self.openEP()
        print("Testing writes")
        dataSent = self.write(ep)
        print()
        print("Testing reads")
        dataRecv = self.read(ep)
        ep.close().wait()
        assert dataSent == dataRecv

    def write_read_many(self, num_msgs=50):
        ep = self.openEP()
        print("Testing writes")
        dataSent = list()
        for _ in range(num_msgs):
            dataSent.append(self.write(ep))
        print()
        print("Testing reads")
        dataRecv = list()
        for _ in range(num_msgs):
            dataRecv.append(self.read(ep))
        ep.close().wait()
        assert dataSent == dataRecv
