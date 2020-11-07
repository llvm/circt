@0xe642127a31681ef6;

# The primary interface exposed by an ESI cosim simulation.
interface CosimDpiServer {
  # List all the registered endpoints.
  list @0 () -> (ifaces :List(EsiDpiInterfaceDesc));
  # Open one of them. Specify both the send and recv data types if want type
  # safety and your language supports it.
  open @1 [S, T] (iface :EsiDpiInterfaceDesc) -> (iface :EsiDpiEndpoint(S, T));
}

# Description of a registered endpoint.
struct EsiDpiInterfaceDesc {
  # Capn'Proto ID of the struct type being sent _to_ the simulator.
  sendTypeID @0 :UInt64;
  # Capn'Proto ID of the struct type being sent _from_ the simulator.
  recvTypeID @1 :UInt64;
  # Numerical identifier of the endpoint. Defined in the design.
  endpointID @2 :Int32;
}

# Interactions with an open endpoint. Optionally typed.
interface EsiDpiEndpoint(SendMsgType, RecvMsgType) {
  # Send a message to the endpoint.
  send @0 (msg :SendMsgType);
  # Recieve a message from the endpoint. Non-blocking.
  recv @1 (block :Bool = true) -> (hasData :Bool, resp :RecvMsgType);
  # Close the connect to this endpoint.
  close @2 ();
}

# A struct for untyped access to an endpoint.
struct UntypedData @0xac6e64291027d47a {
  data @0 :Data;
}
