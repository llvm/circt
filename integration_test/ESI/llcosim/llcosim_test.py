import capnp
import time


def test(schemaPath, hostPort):
  """Load the schema and connect to the RPC server"""
  schema = capnp.load(schemaPath)
  rpc_client = capnp.TwoPartyClient(hostPort)
  cosim = rpc_client.bootstrap().cast_as(schema.LLCosimServer)

  cosim.test().wait()
