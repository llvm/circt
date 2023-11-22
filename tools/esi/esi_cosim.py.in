import capnp
import time


class CosimBase:
  """Provides a base class for cosim tests"""

  def __init__(self, schemaPath, hostPort):
    """Load the schema and connect to the RPC server"""
    self.schema = capnp.load(schemaPath)
    self.rpc_client = capnp.TwoPartyClient(hostPort)
    self.cosim = self.rpc_client.bootstrap().cast_as(self.schema.CosimDpiServer)

  def list(self):
    """List the available interfaces"""
    return self.cosim.list().wait().ifaces

  def openEP(self, epid: str, from_host_type=None, to_host_type=None):
    """Open the endpoint, optionally checking the send and recieve types"""
    ifaces = self.cosim.list().wait().ifaces
    for iface in ifaces:
      if iface.endpointID == epid:
        # Optionally check that the type IDs match.
        if from_host_type is not None:
          assert (iface.fromHostType == from_host_type)
        if to_host_type is not None:
          assert (iface.toHostType == to_host_type)

        openResp = self.cosim.open(iface).wait()
        assert openResp.endpoint is not None
        return openResp.endpoint
    assert False, f"Could not find specified EndpointID '{epid}'"

  def readMsg(self, ep):
    """Cosim doesn't currently support blocking reads. Implement a blocking
           read via polling."""
    while True:
      recvResp = ep.recvToHost().wait()
      if recvResp.hasData:
        break
      else:
        time.sleep(0.01)
    assert recvResp.resp is not None
    return recvResp.resp


class LowLevel:

  def __init__(self, schemaPath, hostPort):
    """Load the schema and connect to the RPC server"""
    self.schema = capnp.load(schemaPath)
    self.rpc_client = capnp.TwoPartyClient(hostPort)
    self.cosim = self.rpc_client.bootstrap().cast_as(self.schema.CosimDpiServer)
    self.low = self.cosim.openLowLevel().wait().lowLevel
