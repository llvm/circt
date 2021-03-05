#!/usr/bin/python3

import random
import cosim


class BasicSystemTester(cosim.CosimBase):
    """Provides methods to test the 'basic' simulation."""

    def testIntAcc(self, num_msgs):
        ep = self.openEP(1, sendType=self.schema.I32,
                         recvType=self.schema.I32)
        sum = 0
        for _ in range(num_msgs):
            i = random.randint(0, 77)
            sum += i
            print(f"Sending {i}")
            ep.send(self.schema.I32.new_message(i=i))
            result = self.readMsg(ep, self.schema.I32)
            print(f"Got {result}")
            assert (result.i == sum)

    def testVectorSum(self, num_msgs):
        ep = self.openEP(2, sendType=self.schema.ArrayOf2xUi24,
                         recvType=self.schema.ArrayOf4xSi13)
        for _ in range(num_msgs):
            # Since the result is unsigned, we need to make sure the sum is
            # never negative.
            arr = [
                random.randint(-468, 777),
                random.randint(500, 1250),
                random.randint(-468, 777),
                random.randint(500, 1250)
            ]
            print(f"Sending {arr}")
            ep.send(self.schema.ArrayOf4xSi13.new_message(l=arr))
            result = self.readMsg(ep, self.schema.ArrayOf2xUi24)
            print(f"Got {result}")
            assert (result.l[0] == arr[0] + arr[1])
            assert (result.l[1] == arr[2] + arr[3])

    def testStruct(self, num_msgs):
        ep = self.openEP(3, sendType=self.schema.Struct13922113893393513056,
                         recvType=self.schema.Struct13922113893393513056)
        for _ in range(num_msgs):
            # Since the result is unsigned, we need to make sure the sum is
            # never negative.
            blob = [random.randint(0, 255) for x in range(32)]
            print(f"Sending data {blob}")
            ep.send(self.schema.Struct13922113893393513056.new_message(
                encrypted=True, compressionLevel=5, blob=blob))
            result = self.readMsg(ep, self.schema.Struct13922113893393513056)
            print(f"Got {result}")
