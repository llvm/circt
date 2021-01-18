#!/usr/bin/python3

import random
import cosim


class BasicSystemTester(cosim.CosimBase):
    """Provides methods to test the 'basic' simulation."""

    def testIntAcc(self, num_msgs):
        ep = self.openEP(sendType=self.schema.I32,
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
