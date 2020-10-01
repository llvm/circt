[Back to table of contents](index.md#Table-of-contents)

# ESI Global Services

**This section not fully fleshed out.** It's quite difficult to understand as
a result. Sorry.

ESI will also provide access to global resource via **ESI Services**. These
buses define a typed interface and instances are instantiated globally
-- they should be accessible at all levels of the design hierarchy. ESI
services are intended to provide access to truly global resources like PCIe,
shared DMA engines, DRAM, network interfaces, etc. They are generic enough to
be extended to non-physical interfaces like telemetry.

## Service/Client & Directions

ESI services have servers and clients which are roughly analogous to
masters and slaves in traditional buses, but with slightly different
meanings. Whereas master/slave denote who can initiate a transaction,
servers/clients denote the many-to-many or one-to-many relationship.
Specifically, servers are expected to have many clients. Any service or
client can initiate a transaction, depending on the direction specified.

Whether a module implements a server or client must be explicitly declared in
the module declaration. Directions (out/in) of data channels/MMIO are
determined by that explicit declaration and the direction specified in the
service definition. Four different directions are allowed in the service
definition: `ToServer`, `ToServers`, `ToClient`, `ToClients`. The plurals
indicate the messages sent should be broadcast to all servers or client. The
singular form indicates that a specific server or client must be addressed
when sending a message unless the server is single server and the sender is a
client.

## Service Definitions

An ESI service is essentially an interface which modules are required to
implement to connect to a given service. The first attribute which must be
specified is whether multiple masters are allowed via the `single` or
`multi` keywords. The second is a list of MMIO and/or data channels and
associated directions.

## Message Sizes

A `ToServer` data channel is free to block (backpressure) other clients
while one is sending a message. As such, in most cases it is best to
keep messages short. While the **list** type is allowed, it is easy to
abuse.

## Wire Level Signaling

The ESI compiler is free to choose any communication substrate which
implements the required functionality. A traditional bus is not
guaranteed, though the ESI compiler will likely select a traditional bus
like Avalon-MM or AXI AMBA in many cases.

## Examples

```c++
struct DMA_Request { ... }
struct DMA_Response { ... }

service HostMemory single { // Specify a service with one server
    ToServer MMIO<any> Memory; // An untyped MMIO wherein the server
                                 // is the 'slave'
    ToServer DMA_Request DmaReq; // DMA access
    ToClient DMA_Response DmaResp;
}

module NiosProcessor { // A NIOS core allowing memory access
    server HostMemory Mem; // A single master server can be implicitly
                           // instantiated and referred to be the
                           // module's instance name
}
```

```c++
union AllStatusesUnion {
    struct PCIeStatus { ... };
    struct Accelerator1Status { ... };
    ...
}

service Telemetry multi { // Specify a service with multiple servers
    ToClients void RequestStatus; // Any server can request a status update
                                  // from every client
    ToServers AllStatusesUnion Statuses; // Clients can respond to all
                                         // the servers
}

Telemetry GlobalBroadcastTelemetry;
```

```c++
service RoutedNetwork multi { // A service with multiple servers
    ToServer NetworkPacket Send; // Clients must select a server
    ToClient NetworkPacket Recv; // Servers must select a particular client
    ToServer MMIO<RoutingConfig> Config; // A per-server configuration interface
}

RoutedNetwork Net;

module NetTelemetryProcessor { // A module which collects telemetry
                               // and sends it over the network
    server Telemetry TelemetryInput;
    client RoutedNetwork TelemetryOutput;
}

NetTelemetryProcessor netTelemProc (
    .TelemetryInput(GlobalBroadcastTelemetry),
    .TelemetryOutput(Net)
)
```
