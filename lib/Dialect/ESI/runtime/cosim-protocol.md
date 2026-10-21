# ESI cosim wire protocol (v4)

The cosim runtime uses a WebSocket connection with a JSON control plane and
binary data frames. The C++ runtime contains both ends of the protocol
(`RpcClient` in `cpp/lib/backends/RpcClient.cpp`, `RpcServer` in
`cpp/lib/backends/RpcServer.cpp`), but the wire format is documented here so
that an out-of-tree client (e.g. a Python harness) can be written against it.

## Transport

- Single WebSocket connection per client, established at
  `ws://<host>:<port>/esi/cosim/v4`.
- The server allows at most one connected client at a time in v4. When an
  additional client connects, the server first sends an unsolicited control
  frame `{"type":"error","error":{"code":"server_busy","message":...}}` to
  give the client a human-readable reason, then closes the WebSocket with
  code 1013 ("Try Again Later", RFC 6455 §7.4). Clients SHOULD surface that
  error message verbatim and MAY retry after a back-off.
- TLS / `wss://` is not used in v4; both ends are expected to run on the same
  host. The transport already supports `wss://` for future remote use.
- The server writes its bound port to `cosim.cfg` in its working directory
  (format: `port: <number>`), so callers that ask the OS to pick the port
  (`COSIM_PORT` unset, or `RpcServer::run(0)`) can discover it.

## Frame discipline

Each WebSocket frame is one of two kinds. The WebSocket opcode itself
discriminates between them, so no extra envelope or pairing is required.

1. **Control frame** — WebSocket *text* frame containing one JSON object.
   Used for `hello`, `connect`, `subscribe`, `unsubscribe`, their replies,
   and flow-control events.
2. **Data frame** — WebSocket *binary* frame containing exactly one ESI
   message:

   ```
   offset 0  : uint64  channel_id   (little-endian)
   offset 8+ : payload bytes        (length = frame_len - 8)
   ```

   No version, flags, or size fields are encoded in the binary frame. The
   protocol version was agreed during `hello`; the WebSocket frame length is
   self-delimiting.

## JSON objects (control frames)

Every JSON object carries a `type` discriminator. Receivers MUST ignore
unrecognised fields so the protocol can evolve additively under the same
`protocol_version`.

### Request (client → server)

```json
{
  "type":       "request",
  "request_id": <uint64>,
  "method":     "<method-name>",
  "params":     { ... }
}
```

`request_id` is a per-connection, monotonically increasing identifier the
client picks so it can match a response back to the call site. The server
echo-es it unchanged in the matching `response`. Reuse across the lifetime
of a single WebSocket connection is forbidden.

### Response (server → client)

Success:
```json
{ "type": "response", "request_id": <uint64>, "result": { ... } }
```

Error:
```json
{
  "type":       "response",
  "request_id": <uint64>,
  "error":      { "code": "<symbolic>", "message": "<human text>" }
}
```

Symbolic error codes used in v4:

| Code                  | Meaning                                          |
| --------------------- | ------------------------------------------------ |
| `unknown_channel`     | Referenced channel id is not in the table.       |
| `wrong_direction`     | Channel has the wrong direction for the method (e.g. `subscribe` to a `to_server` channel). |
| `not_subscribed`      | Tried to unsubscribe from a channel we weren't subscribed to. |
| `protocol_error`      | Malformed frame, missing required field, etc.    |
| `unsupported_version` | `hello` requested a protocol version the server does not speak. |
| `server_busy`         | Server rejected the connection (e.g. another client is already attached). Sent only as an unsolicited `error` frame (see below) right before the server closes the WebSocket. |
| `canceled`            | A `subscribe` still waiting for the simulator side was canceled by `unsubscribe`. |
| `internal`            | Catch-all for server-side failures.              |

### Unsolicited error (server → client)

In rare cases the server needs to report an error that is not the response
to a specific request — most notably when rejecting a freshly opened
connection. In that case the server sends:

```json
{
  "type":  "error",
  "error": { "code": "<symbolic>", "message": "<human text>" }
}
```

... and then closes the WebSocket. The error frame has no `request_id`.
Clients SHOULD surface the `message` field to the user; existing clients
that only recognise `type="response"` will simply ignore it (and still get
the close code, which carries a coarser hint).

## Methods

### Event (either direction)

```json
{ "type": "event", "event": "<event-name>", "params": { ... } }
```

Events are one-way notifications: they carry no `request_id` and are never
answered. Receivers ignore unknown event names. The only event in v4 is
`consumed`; see [Flow control](#flow-control).

### `hello` (client → server, exactly once and first)

```json
{
  "type":       "request",
  "request_id": <uint64>,
  "method":     "hello",
  "params":     { "client_protocol_version": 4 }
}
```

The protocol version must match exactly. If it does not, the server replies
with an `unsupported_version` error. Likewise, the client rejects a reply whose
`protocol_version` differs from its own.

The server holds the response until `RpcServer::setManifest` has been called,
so the client never observes an "empty manifest" state. The reply is:

```json
{
  "type":       "response",
  "request_id": <uint64>,
  "result": {
    "protocol_version":        4,
    "esi_version":             <int32>,
    "compressed_manifest_b64": "<base64>",
    "channels": [
      { "channel_id": 0, "name": "...", "type": "...", "direction": "to_server" },
      { "channel_id": 1, "name": "...", "type": "...", "direction": "to_client" }
    ]
  }
}
```

- `compressed_manifest_b64` is the gzipped ESI manifest, base64-encoded.
- `channels` is the full channel table at the moment the manifest became
  ready. `channel_id`s are server-assigned, dense from 0, stable for the
  lifetime of the connection. The client uses `channel_id` on the wire and
  names in its public C++ API.

### `connect` (client → server)

```json
{ "type": "request", "request_id": N, "method": "connect",
  "params": { "channel_id": <uint64> } }
```

- Channel must have `direction == "to_server"`.
- The server replies `result: {}` once the simulator side of the channel is
  connected and will accept frames, or immediately if it already is. The
  reply may be delayed indefinitely; the server keeps handling other
  requests and frames meanwhile. Clients should not time this request out.
- Clients need not wait for the reply before sending data frames: the
  server holds frames for an unconnected port and they count as in flight,
  so the client is backpressured at its limit. The reply lets a client find
  out that the simulator side is ready. (The C++ runtime sends `connect` from
  `WriteChannelPort::connect()` without waiting, and calls its `onConnected`
  callback on the reply.)
- Errors: `unknown_channel`, `wrong_direction`.

### `subscribe` (client → server)

```json
{ "type": "request", "request_id": N, "method": "subscribe",
  "params": { "channel_id": <uint64>, "max_in_flight": <uint64> } }
```

- Channel must have `direction == "to_client"`.
- `max_in_flight` (optional, default 32) sets the channel's in-flight
  limit; `0` means unbounded. See [Flow control](#flow-control-new-in-v4).
- Like `connect`, the server replies only once the simulator side of the
  channel is connected (immediately if it already is), and the reply may be
  delayed indefinitely. The subscription takes effect when the reply is
  sent. (The C++ runtime doesn't wait for it in `ReadChannelPort::connect()`;
  it calls the `onConnected` callback on the reply.)
- After `result: {}`, the server begins emitting binary data frames for that
  channel id. Any data the server had queued for the channel before
  subscription is flushed first.
- An `unsubscribe` while the reply is still pending cancels the request: it
  fails with `canceled`, and the `unsubscribe` succeeds.
- Errors: `unknown_channel`, `wrong_direction`, `canceled`, and
  `protocol_error` if the channel is already subscribed (or a subscription
  is pending).

### `unsubscribe` (client → server)

```json
{ "type": "request", "request_id": N, "method": "unsubscribe",
  "params": { "channel_id": <uint64> } }
```

- After `result: {}`, the server stops sending frames for that channel.
  Frames the client had received but not consumed are discarded; the
  server stops counting them as in flight, and the channel's in-flight
  limit reverts to the default.
- Errors: `not_subscribed`.

## Data frames

Once a channel is subscribed (server → client) or for any `to_server` channel
(client → server), data is sent as raw WebSocket binary frames with the
eight-byte `channel_id` prefix described above. The byte stream after the
prefix is the unmodified ESI message payload (`MessageData::getBytes()`).

The receiver of a data frame for an unknown or wrong-direction channel logs
the event and discards the frame. Frames are acknowledged in batches via
`consumed` events (below), never individually.

## Flow control (new in v4)

Each channel has an *in-flight limit*: the maximum number of frames which may
be between the producer and the final consumer at once. A frame is in flight
from when the producer hands it to the transport until the receiving side
reports it consumed. Both directions use the same `consumed` event to report
this:

```json
{ "type": "event", "event": "consumed",
  "params": { "channels": [ { "channel_id": <uint64>, "count": <uint64> }, ... ] } }
```

Each entry means `count` more frames on `channel_id` are no longer in flight.
Senders batch entries for several channels into one event; they never send
an entry with `count == 0`.

Usually a frame leaves flight because the final consumer took it. It also
leaves flight when the consuming port discards it, which happens only when
that port is disconnected while holding frames. Frames are never discarded
for want of a consumer: the producer is backpressured instead. The two
directions behave the same way:

| Situation | `to_server` (client → simulation) | `to_client` (simulation → client) |
| --- | --- | --- |
| Consumer not connected yet | Server holds frames until the simulator port connects | Server holds messages until the client subscribes |
| Producer at the limit | `tryWrite()` fails, `write()` blocks | `cosim_ep_tryput` reports full; `DataInReady` deasserts |
| Consumer disconnects | Frames the simulator port held are discarded; the server reports them `consumed` | Frames the client held are discarded; the server stops counting them (`unsubscribe`) |
| Frames still in transit | Server holds them until the port reconnects | Server keeps queued messages for the next subscriber |

### `to_server` channels (client → simulation)

- The client chooses the limit (the C++ runtime uses the write port's
  `ConnectOptions::bufferSize`, defaulting to 16; `0` means unbounded).
- The client counts frames it has sent but which have not been reported
  consumed. It sends a frame only if that count is below the limit.
  `tryWrite()` fails and `write()` blocks while the limit is reached.
- The server sends `consumed` when the simulation takes a frame (the
  `cosim_ep_tryget` DPI call), or when the simulator port discards frames
  on disconnect.
- The server does not enforce the limit; it trusts the client. A client
  MAY admit a multi-frame message while it is below the limit, briefly
  overshooting it.

### `to_client` channels (simulation → client)

- The limit comes from `subscribe`'s `max_in_flight` (the C++ runtime uses
  the read port's `ConnectOptions::bufferSize`, defaulting to 32). Without a
  subscriber, the default of 32 applies.
- The server counts frames queued for sending plus frames sent but not yet
  reported consumed. While that count is at the limit, the simulator-side
  port refuses new messages: `cosim_ep_tryput` returns a positive value and
  `Cosim_Endpoint_ToHost` deasserts `DataInReady` until there is room.
- The client sends `consumed` once its read port accepts frames (the
  user callback returns true, or the polling queue takes them). It does not
  report frames it discards: those for a channel it has unsubscribed from, or
  frames left in its queue on `unsubscribe`. The server instead stops counting
  them, as below.
- On `unsubscribe` or disconnect, the server forgets frames sent but not
  acknowledged. It ignores `consumed` counts beyond the number in flight
  (for example, a late `consumed` which crossed an `unsubscribe`).

## Lifecycle

- **Open:** TCP + WS handshake → client sends `hello` → server replies once
  `setManifest` has been called.
- **Steady state:** bidirectional binary data frames, `consumed` events,
  and occasional `connect`/`subscribe`/`unsubscribe` requests.
- **Shutdown:** either side sends a WebSocket Close frame. The server treats a
  client close as implicit unsubscribe-all + cleanup. The client treats a
  server close as connection loss: pending request promises fail, writers
  blocked at an in-flight limit throw, and registered read callbacks are
  signaled to terminate.

## Reserved / future

These are *not* part of v4; they are listed here so a v4 implementation can
ignore them safely and a future protocol revision can add them without
breaking the channel-id namespace.

- Per-message flags on data frames: would be added under `protocol_version`
  bump as either a prefixed byte or an explicit JSON header.
- TLS (`wss://`) and bearer-token auth via `hello` params.
- Multi-client servers.



# History of cosim RPC protocol versions

## v1 -- Cap'nProto RPC

We initially used Cap'nProto (capnp.org) for the RPC protocol, which provided a
schema language and codegen for multiple languages. It also was a reasonably
small (compared to v2) compile time dependency. The downside was that it uses an
obscure runtime library (libkj) with strange (though logical) semantics / API
and was (at the time) largely undocumented. As a result, implementing changes
which should have been simple took days. Anything which didn't use libkj's async
methodology was not simple to implement and always felt like a hack.

## v2 -- gRPC

We switched to gRPC for v2, which was a more familiar and well-documented RPC
framework. The decision was made due largely to the documentation, larger user
base, and standard threading model. During this transition, we also abstracted
the RPC layer behind an `RpcClient` / `RpcServer` interface, which allowed us to
keep the same public C++ API and isolate the gRPC dependency in the
implementation. This would turn out to be a good decision, as we ended up
switching to the v3 protocol.

gRPC was _waaaay_ too large of a compile-time dependency for our need. Builds
ended up taking an additional 2+ hours to compile gRPC and its dependencies.
Also, it pulls in abseil which had conflicts with other parts of CIRCT which has
dependencies on abseil but different versions.

## v3 -- Custom WebSocket + JSON protocol

Should be much quicker to compile but the downside is that we have to
maintain the protocol ourselves. This, however, has turned out to be minimal. It
also has the advantage of being (theoretically) easier to debug as we own all of
the code.

## v4 -- Connection readiness and flow control

This one. Same transport and framing as v3. Adds the `connect` method, holds
`subscribe` replies until the simulator side of the channel is connected,
holds frames for unconnected simulator ports rather than dropping them, and
checks the protocol version exactly in `hello`. Also adds credit-based flow
control in both directions via `consumed` events and the `max_in_flight`
subscribe parameter. In v3 every queue was unbounded: the client could flood
the simulator, and cosim never deasserted ready to the accelerator.
