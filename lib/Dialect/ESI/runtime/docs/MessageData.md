# SegmentedMessageData: multi-segment serialization and deserialization

Goals:
- Support generated types with variable-length lists (header + list data)
  without flattening into a single contiguous buffer before sending.
- Support matching read-side deserialization for those types without
  requiring every raw message to map 1:1 onto a typed object.
- Keep existing flat `MessageData` entry points available for compatibility.
- Unify backend/client internals on an owning `SegmentedMessageData` path so
  read retries can preserve message ownership and identity.
- Let backends handle multi-segment messages efficiently (scatter-gather,
  chunked DMA) or fall back to flattening.

## Background

The existing `MessageData` is a concrete, value-type class wrapping a
`std::vector<uint8_t>`. It is used by value throughout the public flat APIs
(`write(const MessageData &)`, `read(MessageData &)`, flat callbacks,
futures). Replacing it outright with an abstract segmented type would require
rewriting every port, backend, and user-facing API.

Instead, `SegmentedMessageData` becomes the common abstract base and
`MessageData` becomes its concrete one-segment implementation. Generated types
that need multiple segments (e.g. a fixed-size header followed by a
variable-length list) also subclass `SegmentedMessageData`. This keeps the
public flat APIs available while letting the backend/client internals move a
single owning message type through retry-capable paths.

That same abstraction can serve as the raw input to a read-side
`TypeDeserializer`. The code generator emits a `TypeDeserializer` class for
each user-defined type that needs one, and the runtime creates one stateful
instance per connected read port. That instance accepts owned segmented
messages, allocates typed objects once they are fully available, and pushes
those objects to clients via a callback.

Importantly, the callback contract must preserve the existing read-port
meaning of backpressure: when the callee says "not accepted yet", the caller
must still retain ownership so it can retry later.

## Design

`ReadChannelPort` callback mode now uses an owning segmented callback path as
its canonical internal representation:

`std::function<bool(std::unique_ptr<SegmentedMessageData> &)>`

That lets backends preserve message ownership and identity across retries.
Existing flat callbacks and polling reads remain available as adapters layered
on top of that segmented path.

A separate `Cursor` class tracks consumption position across segments. Backends
that need to resume partial writes store both a `unique_ptr<SegmentedMessageData>`
and a `Cursor` — keeping the cursor state out of the generated type so its
layout matches the hardware wire format exactly.

On the read side, the same segmentation abstraction is fed into a nested
`TypeDeserializer` class under the generated user type. The deserializer owns
all incremental parse state. Each incoming raw message may produce zero, one,
or many typed output objects, so the interface is callback-driven rather than
`read()`-style value-returning. The raw message itself is passed by
`std::unique_ptr<SegmentedMessageData> &` so the deserializer can consume it,
move storage out of it, or leave it untouched for retry.

## API Details

```c++
  /// A contiguous, non-owning view of bytes.
  /// SegmentedMessageData subclasses own the storage; Segment just points
  /// into it. Valid only while the owning SegmentedMessageData is alive.
  struct Segment {
    const uint8_t *data;
    size_t size;
    std::span<const uint8_t> span() const { return {data, size}; }
    bool empty() const { return size == 0; }
  };

  /// Abstract multi-segment message. Generated types subclass this to
  /// expose header + list segments without flattening.
  ///
  /// Subclasses MUST own all data that their segments point to. This is
  /// required because the write API takes ownership
  /// (unique_ptr<SegmentedMessageData>) and a backend may hold the
  /// message across async boundaries / partial writes.
  class SegmentedMessageData {
  public:
    virtual ~SegmentedMessageData() = default;

    // --- Segment access ---
    virtual size_t numSegments() const = 0;
    virtual Segment segment(size_t idx) const = 0;

    // --- Convenience ---
    size_t totalSize() const;
    bool empty() const;

    /// Flatten all segments into a standard MessageData.
    virtual MessageData toMessageData() const;
  };

  /// A concrete flat message backed by a single vector of bytes.
  class MessageData : public SegmentedMessageData {
  public:
    // Existing vector-backed API unchanged.

    size_t numSegments() const override { return 1; }
    Segment segment(size_t idx) const override;
    MessageData toMessageData() const override { return *this; }
  };

  /// Cursor for incremental consumption of a SegmentedMessageData.
  /// Tracks position across segment boundaries. Backends store this
  /// alongside the unique_ptr<SegmentedMessageData> for partial writes.
  ///
  /// Deliberately a separate class (not embedded in SegmentedMessageData)
  /// so that generated types have no hidden members — their layout
  /// matches the hardware wire format exactly.
  class SegmentedMessageDataCursor {
  public:
    SegmentedMessageDataCursor(const SegmentedMessageData &msg)
      : msg_(msg) {}

    /// Contiguous span from current position to end of current segment.
    std::span<const uint8_t> remaining() const;

    /// Advance by `n` bytes, crossing segment boundaries as needed.
    void advance(size_t n);

    /// True when all segments have been consumed.
    bool done() const;

    /// Reset to the beginning.
    void reset();

  private:
    const SegmentedMessageData &msg_;
    size_t segIdx_ = 0;
    size_t offset_ = 0;
  };

```

## Port API additions

```c++
  class ReadChannelPort : public ChannelPort {
  public:
    using ReadCallback =
        std::function<bool(std::unique_ptr<SegmentedMessageData> &)>;
    using FlatReadCallback = std::function<bool(MessageData)>;

    virtual void connect(ReadCallback callback,
                         const ConnectOptions &options = {});
    virtual void connect(FlatReadCallback callback,
                         const ConnectOptions &options = {});

    // Polling `connect()`, `readAsync()`, and `read(MessageData &)` remain.
  };

  class WriteChannelPort : public ChannelPort {
    // ... existing API unchanged ...

    /// Write a multi-segment message. Takes ownership so the backend can
    /// hold the message across partial writes / async completions.
    /// Default flattens and calls writeImpl().
    /// Backends override for scatter-gather / chunked-DMA support.
    virtual void write(std::unique_ptr<SegmentedMessageData> msg) {
      write(msg->toMessageData());
    }
  };

```

Backends that do not override segmented write handling get correct behavior
automatically via flattening. Backends that *do* override it receive sole
ownership of the message and can store the `unique_ptr` to resume partial
writes later, alongside a `SegmentedMessageDataCursor` to track progress.

On the read side, the owning segmented callback is now the canonical internal
path. A backend can retain and retry the same `SegmentedMessageData` object
until the callback accepts it. Existing flat callbacks and polling reads still
work, but they are adapters layered on top of the segmented ownership path.

## Type serialization (write side)

The generated type *itself* subclasses `SegmentedMessageData`. Its segments
point directly at its own member data — no wrapper class, no intermediate
copies, no packing step. The struct IS the message.

Since the generated code controls the struct layout, it emits a packed
header sub-struct whose in-memory layout matches the wire format exactly.
Segments point directly at the struct's own fields.

A generated type with a variable-length list looks like:

```c++
  struct SampleBatch : public SegmentedMessageData {
    // Packed POD header — in-memory layout matches the wire format.
#pragma pack(push, 1)
    struct Header {
      uint32_t batchId;
      uint16_t status;
    };
#pragma pack(pop)

    Header header;
    std::vector<uint32_t> samples;  // variable-length

    // --- SegmentedMessageData interface ---

    size_t numSegments() const override { return 2; }
    Segment segment(size_t i) const override {
      if (i == 0)
        return {reinterpret_cast<const uint8_t *>(&header), sizeof(Header)};
      return {reinterpret_cast<const uint8_t *>(samples.data()),
              samples.size() * sizeof(uint32_t)};
    }
  };
```

The wire format is: [packed header bytes] [list element bytes...], with list
length either implicit (consumes the rest of the message) or encoded in a
header field.

### How it works

The generated `SampleBatch` struct:
- Subclasses `SegmentedMessageData` directly.
- Contains a packed anonymous struct for the header fields whose in-memory
  layout matches the wire format (no padding between fields).
- `segment(0)` points at `&header` — the header fields in place.
- `segment(1)` points at `samples.data()` — directly into the vector's
  storage.
- No packing step, no intermediate buffer. Every segment points at data
  that the struct already owns.

Since the struct owns both `header` and `samples`, all segments point at
data owned by `this`. When write takes a `unique_ptr<SegmentedMessageData>`,
the `SampleBatch` and all its data stay alive as long as the backend needs
them.

### Cost analysis

Moving a `SampleBatch` into `make_unique`: the `samples` vector is
move-constructed
(pointer swap — zero-copy for potentially megabytes of list data). The
packed header is trivially copied (6 bytes). No intermediate wrapper, no
extra allocations beyond the single `unique_ptr`.

### Usage at the call site

```c++
  SampleBatch batch;
  batch.header.batchId = 42;
  batch.header.status = 0x01;
  batch.samples = buildSamples();  // potentially large

  // Move batch onto the heap and write. batch.samples is empty after this.
  port.write(std::make_unique<SampleBatch>(std::move(batch)));
```

If the backend doesn't override `write(unique_ptr<SegmentedMessageData>)`,
the default flattens via `toMessageData()` and frees the message. If the
backend supports scatter-gather, it stores the `unique_ptr` and sends each
segment as a separate DMA descriptor. If a partial write occurs, the
backend keeps the `unique_ptr` and uses the embedded cursor to resume.

### TypedWritePort integration

```c++
  template <>
  class TypedWritePort<SampleBatch> {
    void write(std::unique_ptr<SampleBatch> p) {
      port.write(std::move(p));
    }
  };
```

The typed overload takes ownership via `unique_ptr`, matching the
`WriteChannelPort::write(unique_ptr<SegmentedMessageData>)` overload.
The generated specialization casts the owned `SampleBatch` up to
`SegmentedMessageData` implicitly on the move.

## Type de-serialization (read side)

The read-side counterpart is a nested `TypeDeserializer` generated under the
user-defined type. It is responsible for:

- Accepting owned raw segmented messages from `ReadChannelPort`.
- Owning any incremental parse state required across callbacks.
- Allocating typed objects once enough raw data has been assembled.
- Transferring ownership of completed objects to clients via
  `std::unique_ptr`.

The crucial difference from the write side is that raw input messages are not
required to map 1:1 to typed outputs. A single input message may yield no
objects yet, exactly one object, or multiple objects. Likewise, one typed
object may require bytes from multiple input messages.

That means the deserializer should not return a single object from its input
method. Instead, it should push completed objects to a client-supplied output
callback.

It also means both callbacks need the same ownership convention. Existing
`ReadChannelPort` callbacks use `false` to mean "this message was not
consumed; retry it later". For both the raw segmented input callback and the
typed output callback, that requires the callee to receive a
`std::unique_ptr<T> &` rather than a moved value.

### Proposed generated API

```c++
  struct SampleBatch : public SegmentedMessageData {
    // ... existing write-side fields and segment() implementation ...

    class TypeDeserializer : public QueuedDecodeTypeDeserializer<SampleBatch> {
    public:
      using Base = QueuedDecodeTypeDeserializer<SampleBatch>;
      using OutputCallback = Base::OutputCallback;
      using DecodedOutputs = Base::DecodedOutputs;

      explicit TypeDeserializer(OutputCallback output)
          : Base(std::move(output)) {}

    private:
      DecodedOutputs decode(std::unique_ptr<SegmentedMessageData> &msg)
          override;
      // Generated state for partial header/list accumulation goes here.
    };
  };
```

Nesting the class under `SampleBatch` gives `TypedReadPort<SampleBatch>` a
stable,
discoverable hook: it can look for `SampleBatch::TypeDeserializer` and use it
automatically.

### Why the deserializer allocates objects

For list-containing or otherwise variable-size types, the final object size is
not known until parsing completes. The deserializer therefore owns the object
construction step. Once an object is complete, it transfers ownership via
`std::unique_ptr<SampleBatch>` to avoid copies and to make lifetime explicit.

When downstream is not ready after a raw message has already been consumed, the
deserializer keeps completed objects in a FIFO pending-output queue and rejects
subsequent input messages until that queue has drained.

Most stateful deserializers need that retry-and-queue behavior but do not need
to own the `push()` control flow themselves. `QueuedDecodeTypeDeserializer<T>`
is provided as a small optional base class for that pattern: derived
deserializers only implement `decode(msg)` and return the completed typed
objects produced by that raw message.

`poke()` retries delivery from the pending-output queue without requiring a new
raw input message. This gives higher layers a way to re-attempt delivery when
their own buffering state changes.

If the deserializer cannot accept a new raw message at all, it returns false
from `push()` without consuming `msg`, leaving the caller in possession of the
same message for retry.

### What the push return means

For `QueuedDecodeTypeDeserializer<T>`, `decode(msg)` is responsible for
consuming the raw message and returns zero, one, or many completed typed
objects. `push()` returns `true` once `decode(msg)` has consumed the raw
message, even if some of those typed outputs had to be parked in the pending
queue for later delivery.

`decode(msg)` may also keep partial logical-frame bytes in its own state when
raw message boundaries do not align with logical decode boundaries. Raw input
messages should not be assumed to end on typed frame boundaries.

`false` means the deserializer did not accept the raw message. For
`QueuedDecodeTypeDeserializer<T>`, that only happens when previously decoded
outputs are still pending, in which case `push()` does not call `decode(msg)`
and leaves `msg` owning the original message so the caller can retry later.

### What the callback return means

`true` means the callback has accepted the object. In the usual case, it does
that by moving from the `std::unique_ptr<SampleBatch> &` argument.

`false` means the callback did not accept the object yet. The deserializer
retains that object and any later decoded objects in its pending-output queue
and should not consume
more raw input until the output is accepted.

### Why the interface is callback-based

The existing polling API (`read()`, `readAsync()`) assumes one queued raw
message corresponds to one typed value. That assumption does not hold once a
deserializer may buffer input or emit multiple objects per input message.

For that reason, the `TypeDeserializer` itself remains callback-oriented: it
accepts raw messages and emits typed objects through callbacks while
maintaining per-port decode state. `TypedReadPort` can still provide `read()`
and `readAsync()` by maintaining a bounded buffer of `std::unique_ptr<T>`
objects plus a promise queue, using the same basic
fulfillment pattern as `ReadChannelPort`. The exact storage choice and buffer
depth can be implementation-defined, but it should not be unbounded. When
buffer space becomes available or a new promise is installed, `TypedReadPort`
calls `poke()` to retry delivery of any blocked output. So the callback-only
restriction applies to the internal deserializer interface, not to
`TypedReadPort` as a whole.

## TypedReadPort integration

`TypedReadPort<T>` would use a deserializer-based path for both simple and
complex types.

For ordinary scalar/POD-like types, `TypedReadPort<T>` can use an internal
`PODTypeDeserializer<T>` which simply calls `fromMessageData<T>()` and forwards
the typed value to the callback. Because that conversion is 1:1 and does not
need to consume the raw message before the callback accepts it,
`PODTypeDeserializer<T>` does not need a retry slot.

For generated segmented types, `TypedReadPort<T>` uses the nested
`T::TypeDeserializer` and:

- Detects `T::TypeDeserializer` at compile time.
- Instantiates one deserializer per `connect()` call.
- Passes the user callback into the deserializer constructor.
- Connects the raw `ReadChannelPort` using the new segmented callback overload.
- Lets the deserializer manage any 0:1, 1:N, or N:1 relationship between raw
  messages and typed objects.
- For `read()` / `readAsync()`, buffers `std::unique_ptr<T>` values in a
  bounded, implementation-defined queue so raw input can be drained
  independently of user consumption.
- For polling mode, calls `poke()` whenever buffer space becomes available or a
  waiting promise is added, so a previously blocked output can be retried.
- For callback mode, blocked output delivery would need `poke()` to be driven
  periodically by the existing background worker / poll machinery.

Conceptually, the typed hookup looks like:

```c++
  template <typename T>
  class TypedReadPort {
  public:
    void connect(OutputCallback<T> callback,
                 const ChannelPort::ConnectOptions &opts = {}) {
      using Deserializer = DeserializerFor<T>;
      auto deserializer = std::make_shared<Deserializer>(std::move(callback));
      inner->connect(
          [deserializer](std::unique_ptr<SegmentedMessageData> &msg) -> bool {
            return deserializer->push(msg);
          },
          opts);
    }
  };
```

Using `shared_ptr` here is only to keep the deserializer alive for as long as
the read callback remains registered. For simple types, `DeserializerFor<T>`
can be `PODTypeDeserializer<T>`. For complex generated types, it is the nested
`T::TypeDeserializer`. The important common API contract is
`push(std::unique_ptr<SegmentedMessageData> &)`.

For polling mode, `TypedReadPort` uses an internal output callback which:

- Immediately fulfills the oldest waiting promise, if one exists.
- Otherwise stores the object in a bounded typed buffer.
- Returns `false` if that buffer is full, so the deserializer keeps the current
  and any later decoded objects in its pending-output queue and stops
  accepting new raw input.

When `read()` / `readAsync()` consumes buffered output, or when a new promise
is registered, `TypedReadPort` calls `poke()` to re-attempt delivery of any
queued output objects.

`PODTypeDeserializer<T>` does not need `QueuedDecodeTypeDeserializer<T>` or
`poke()` for
correctness: if the output callback cannot accept the decoded object, it can
simply return false from `push()` without consuming the raw message.

For callback mode, `TypedReadPort` does not currently have its own `poll()`
entry point and is not yet hooked into the existing background worker thread.
TODO: when implementing this design, wire callback-mode `poke()` retries into
the existing periodic poll/background-worker path so blocked typed output
delivery is retried even when no new raw input arrives.

## Proposed implementation plan

1. Treat `SegmentedMessageData` as the common ordered-byte abstraction for
  both write-side serialization and read-side deserialization.
2. Make `MessageData` the concrete flat, one-segment implementation of
  `SegmentedMessageData`, so flat and segmented messages share one internal
  ownership model.
3. Switch `ReadChannelPort` internals to an owning
  `std::function<bool(std::unique_ptr<SegmentedMessageData> &)>` callback,
  while keeping flat callbacks and polling reads as adapters layered on top.
4. Define a nested `T::TypeDeserializer` convention for generated/user-defined
  types that need stateful or variable-length decoding, and add an internal
  `PODTypeDeserializer<T>` for ordinary 1:1 scalar/POD types.
5. Extend `TypedReadPort<T>` to choose the appropriate deserializer,
  wire it into the segmented callback path, and expose a
  `std::function<bool(std::unique_ptr<T> &)>` client callback for complex
  types.
6. Add `TypeDeserializer::poke()` for deserializers which may need to retry
  delivery of a blocked output object without requiring new raw input.
7. For deserializer-backed `TypedReadPort<T>`, implement `read()` and
  `readAsync()` using a bounded `std::unique_ptr<T>` buffer plus a promise
  queue.
8. TODO: hook callback-mode `poke()` retries into the existing periodic
  poll/background-worker machinery, since `TypedReadPort` does not currently
  expose its own `poll()` entry point.
9. Add tests covering POD/scalar deserialization via `PODTypeDeserializer<T>`,
  zero-output, one-output, many-output, cross-message partial decode,
  backpressure propagation, `poke()` retry behavior, typed polling with a
  bounded output buffer, and callback-mode retry once the poll hookup exists.

## Examples

### Sending through existing port APIs (no backend changes needed)

```c++
  SampleBatch batch = buildSampleBatch();
  // Flatten to MessageData and write through existing API.
  port.write(batch.toMessageData());
```

### Sending through the segmented overload (zero-copy list)

```c++
  SampleBatch batch = buildSampleBatch();
  port.write(std::make_unique<SampleBatch>(std::move(batch)));
  // batch.samples now empty
```

### Raw ReadChannelPort hookup with a deserializer

```c++
  SampleBatch::TypeDeserializer deserializer(
      [](std::unique_ptr<SampleBatch> &batch) -> bool {
        consumeBatch(std::move(batch));
        return true;
      });

  readPort.connect(
      [&deserializer](std::unique_ptr<SegmentedMessageData> &msg) -> bool {
        return deserializer.push(msg);
      });
```

### TypedReadPort hookup for a generated type

```c++
  TypedReadPort<SampleBatch> typed(readPort);
  typed.connect([](std::unique_ptr<SampleBatch> &batch) -> bool {
    handleBatch(std::move(batch));
    return true;
  });
```

### Backend scatter-gather override (synchronous)

```c++
  // This makes the assumption that all of the segments are mapped into
  // accelerator address space and all at the same address... which isn't
  // generally true.
  // TODO: add an API to get the accelerator address for a segment, so backends
  // can support truly general scatter-gather.
  class MyBackendWritePort : public WriteChannelPort {
    void write(std::unique_ptr<SegmentedMessageData> msg) override {
      // Scatter-gather: send each segment as a separate DMA descriptor.
      for (size_t i = 0; i < msg->numSegments(); ++i) {
        auto seg = msg->segment(i);
        dma_enqueue(seg.data, seg.size);
      }
      dma_submit();
      // msg freed on return — all data was sent.
    }
  };
```

### Backend storing message for partial / async DMA

```c++
  class MyBackendWritePort : public WriteChannelPort {
    /// This isn't thread-safe and doesn't support async writes, but illustrates
    /// the basic idea of storing the message and cursor for resumption later.
    std::unique_ptr<SegmentedMessageData> pending;
    std::optional<SegmentedMessageDataCursor> cursor;

    void write(std::unique_ptr<SegmentedMessageData> msg) override {
      pending = std::move(msg);
      cursor.emplace(*pending);
      drainPending();
    }

    // Called from event loop / interrupt handler.
    void drainPending() {
      while (pending && !cursor->done()) {
        auto chunk = cursor->remaining();
        size_t sent = dma_write(chunk.data(),
                                std::min(chunk.size(), bufferSpace));
        if (sent == 0)
          return;  // buffer full — resume later
        cursor->advance(sent);
      }
      cursor.reset();
      pending.reset();  // all done — free the message
    }
  };
```

## What does NOT change

- `MessageData` remains a concrete value type wrapping `std::vector<uint8_t>`.
- The public flat APIs remain: `write(const MessageData &)`,
  `connect(std::function<bool(MessageData)>)`, `readAsync()`, and
  `read(MessageData &)`. They now sit on top of the segmented path.
- `WriteChannelPort::writeImpl()` remains flat.
- All existing backends continue to work via the default flattening write path
  and flat callback/polling adapters.
- All existing user code that constructs `MessageData` directly: unchanged.
- `TypedWritePort` / `TypedReadPort` for scalar types: unchanged.
- The existing polling/raw `MessageData` read path: unchanged.
- Generated list-containing types opt in by defining a nested
  `TypeDeserializer`.
