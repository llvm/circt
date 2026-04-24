# SegmentedMessageData: multi-segment message support

Goals:
- Support generated types with variable-length lists (header + list data)
  without flattening into a single contiguous buffer before sending.
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

## Design

`SegmentedMessageData` is a pure abstract class representing an ordered
sequence of contiguous byte spans ("segments"). `MessageData` is the flat,
single-segment implementation. Generated types subclass it to expose their
header and list-body segments without copying them into a single buffer.

A virtual `toMessageData()` method flattens segmented subclasses into a
standard flat `MessageData`. `MessageData` overrides it trivially by returning
itself.

For backends that support scatter-gather, chunked DMA, or avoid a memcpy to
write directly into its buffer, `WriteChannelPort` gains an optional virtual
`write()` method overload that accepts a `SegmentedMessageData` directly. The
default implementation calls `toMessageData()` and forwards to the existing
`writeImpl()`, so all existing backends keep working without modification.

`ReadChannelPort` likewise unifies its callback-mode internals around an owning
`std::unique_ptr<SegmentedMessageData> &` callback. This lets a backend keep
the exact same message object alive across `false` retries. Legacy flat
callbacks and polling reads remain available as adapters layered on top of that
segmented callback path.

A separate `Cursor` class tracks consumption position across segments. Backends
that need to resume partial writes store both a `unique_ptr<SegmentedMessageData>`
and a `Cursor` — keeping the cursor state out of the generated type so its
layout matches the hardware wire format exactly.

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
  struct Packet : public SegmentedMessageData {
    // Packed POD header — in-memory layout matches the wire format.
#pragma pack(push, 1)
    struct Header {
      uint32_t id;
      uint16_t flags;
    };
#pragma pack(pop)

    Header header;
    std::vector<uint32_t> items;  // variable-length

    // --- SegmentedMessageData interface ---

    size_t numSegments() const override { return 2; }
    Segment segment(size_t i) const override {
      if (i == 0)
        return {reinterpret_cast<const uint8_t *>(&header), sizeof(Header)};
      return {reinterpret_cast<const uint8_t *>(items.data()),
              items.size() * sizeof(uint32_t)};
    }
  };
```

The wire format is: [packed header bytes] [list element bytes...], with list
length either implicit (consumes the rest of the message) or encoded in a
header field.

### How it works

The generated `Packet` struct:
- Subclasses `SegmentedMessageData` directly.
- Contains a packed anonymous struct for the header fields whose in-memory
  layout matches the wire format (no padding between fields).
- `segment(0)` points at `&header` — the header fields in place.
- `segment(1)` points at `items.data()` — directly into the vector's
  storage.
- No packing step, no intermediate buffer. Every segment points at data
  that the struct already owns.

Since the struct owns both `header` and `items`, all segments point at
data owned by `this`. When write takes a `unique_ptr<SegmentedMessageData>`,
the Packet and all its data stay alive as long as the backend needs them.

### Cost analysis

Moving a `Packet` into `make_unique`: the `items` vector is move-constructed
(pointer swap — zero-copy for potentially megabytes of list data). The
packed header is trivially copied (6 bytes). No intermediate wrapper, no
extra allocations beyond the single `unique_ptr`.

### Usage at the call site

```c++
  Packet pkt;
  pkt.header.id = 42;
  pkt.header.flags = 0x01;
  pkt.items = buildItems();  // potentially large

  // Move pkt onto the heap and write. pkt.items is empty after this.
  port.write(std::make_unique<Packet>(std::move(pkt)));
```

If the backend doesn't override `write(unique_ptr<SegmentedMessageData>)`,
the default flattens via `toMessageData()` and frees the message. If the
backend supports scatter-gather, it stores the `unique_ptr` and sends each
segment as a separate DMA descriptor. If a partial write occurs, the
backend keeps the `unique_ptr` and uses the embedded cursor to resume.

### TypedWritePort integration

```c++
  template <>
  class TypedWritePort<Packet> {
    void write(std::unique_ptr<Packet> p) {
      port.write(std::move(p));
    }
  };
```

The typed overload takes ownership via `unique_ptr`, matching the
`WriteChannelPort::write(unique_ptr<SegmentedMessageData>)` overload.
The generated specialization casts the owned `Packet` up to
`SegmentedMessageData` implicitly on the move.

## Examples

### Sending through existing port APIs (no backend changes needed)

```c++
  Packet pkt = buildPacket();
  // Flatten to MessageData and write through existing API.
  port.write(pkt.toMessageData());
```

### Sending through the segmented overload (zero-copy list)

```c++
  Packet pkt = buildPacket();
  port.write(std::make_unique<Packet>(std::move(pkt)));  // pkt.items now empty
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
- All existing user code that constructs `MessageData` directly: unchanged.
- `TypedWritePort` / `TypedReadPort`: unchanged for scalar types.
  Generated specializations for list-containing types opt in.
