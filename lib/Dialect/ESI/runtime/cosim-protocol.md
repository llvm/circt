# ESI cosim wire protocol (v3)

The cosim runtime uses a WebSocket connection with a JSON control plane and
binary data frames. The C++ runtime contains both ends of the protocol
(`RpcClient` in `cpp/lib/backends/RpcClient.cpp`, `RpcServer` in
`cpp/lib/backends/RpcServer.cpp`), but the wire format is documented here so
that an out-of-tree client (e.g. a Python harness) can be written against it.

## Transport

- Single WebSocket connection per client, established at
  `ws://<host>:<port>/esi/cosim/v3`.
- The server allows at most one connected client at a time in v3; additional
  connections receive a WebSocket close with code 1011 (internal error).
- TLS / `wss://` is not used in v3; both ends are expected to run on the same
  host. The transport already supports `wss://` for future remote use.
- The server writes its bound port to `cosim.cfg` in its working directory
  (format: `port: <number>`), so callers that ask the OS to pick the port
  (`COSIM_PORT` unset, or `RpcServer::run(0)`) can discover it.

## Frame discipline

Each WebSocket frame is one of two kinds. The WebSocket opcode itself
discriminates between them, so no extra envelope or pairing is required.

1. **Control frame** — WebSocket *text* frame containing one JSON object.
   Used for `hello`, `subscribe`, `unsubscribe`, and their replies.
2. **Data frame** — WebSocket *binary* frame containing exactly one ESI
   message:

   ```
   offset 0  : uint16  channel_id   (little-endian)
   offset 2+ : payload bytes        (length = frame_len - 2)
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

Symbolic error codes used in v3:

| Code                  | Meaning                                          |
| --------------------- | ------------------------------------------------ |
| `unknown_channel`     | Referenced channel id is not in the table.       |
| `wrong_direction`     | Tried to subscribe to a `to_server` channel.     |
| `not_subscribed`      | Tried to unsubscribe from a channel we weren't subscribed to. |
| `protocol_error`      | Malformed frame, missing required field, etc.    |
| `server_busy`         | Server rejected the connection (e.g. another client is already attached). Sent only as an unsolicited `error` frame (see below) right before the server closes the WebSocket. |
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

### `hello` (client → server, exactly once and first)

```json
{
  "type":       "request",
  "request_id": <uint64>,
  "method":     "hello",
  "params":     { "client_protocol_version": 3 }
}
```

The server holds the response until `RpcServer::setManifest` has been called,
so the client never observes an "empty manifest" state. The reply is:

```json
{
  "type":       "response",
  "request_id": <uint64>,
  "result": {
    "protocol_version":        3,
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

### `subscribe` (client → server)

```json
{ "type": "request", "request_id": N, "method": "subscribe",
  "params": { "channel_id": <uint16> } }
```

- Channel must have `direction == "to_client"`.
- After `result: {}`, the server begins emitting binary data frames for that
  channel id. Any data the server had queued for the channel before
  subscription is flushed first.
- Errors: `unknown_channel`, `wrong_direction`.

### `unsubscribe` (client → server)

```json
{ "type": "request", "request_id": N, "method": "unsubscribe",
  "params": { "channel_id": <uint16> } }
```

- After `result: {}`, the server stops sending frames for that channel.
- Errors: `not_subscribed`.

## Data frames

Once a channel is subscribed (server → client) or for any `to_server` channel
(client → server), data is sent as raw WebSocket binary frames with the
two-byte `channel_id` prefix described above. The byte stream after the
prefix is the unmodified ESI message payload (`MessageData::getBytes()`).

The receiver of a data frame for an unknown or wrong-direction channel logs
the event and discards the frame. There is no per-message ack.

## Lifecycle

- **Open:** TCP + WS handshake → client sends `hello` → server replies once
  `setManifest` has been called.
- **Steady state:** bidirectional binary data frames plus occasional
  `subscribe`/`unsubscribe` requests.
- **Shutdown:** either side sends a WebSocket Close frame. The server treats a
  client close as implicit unsubscribe-all + cleanup. The client treats a
  server close as connection loss: pending request promises fail and
  registered read callbacks are signaled to terminate.

## Reserved / future

These are *not* part of v3; they are listed here so a v3 implementation can
ignore them safely and a future protocol revision can add them without
breaking the channel-id namespace.

- Event push (server → client without a `request_id`): reserved
  `{ "type": "event", "event": "...", "params": { ... } }`.
- Per-message flags on data frames: would be added under `protocol_version`
  bump as either a prefixed byte or an explicit JSON header.
- TLS (`wss://`) and bearer-token auth via `hello` params.
- Multi-client servers.


## Future improvements

### Backpressure

In both directions, we current use unbounded queues to buffer data which could
easily lead to a huge memory usage, especially if the client doesn't subscribe
to a channel. It is also unrealistic since cosim ports never present "not ready"
signals to the accelerator, and when the client sends data it always succeeds.

In the to_client direction, add a mechanism for the RPC server to signal the
"not ready" state of a channel to the cosim RTL module. In the to_server
direction, add a credit-control mechanism to this protocol.


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

This one. Should be much quicker to compile but the downside is that we have to
maintain the protocol ourselves. This, however, has turned out to be minimal. It
also has the advantage of being (theoretically) easier to debug as we own all of
the code.
