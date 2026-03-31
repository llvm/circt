= Message data memory design and usage

Goals:
- Provide a simple interface for users to send and recieve data from the
  accelerator.
- Provide an interface for generated types to serialize and deserialize
  themselves to and from the accelerator.
- Make it efficient, primarily designed to reduce the number of memory copies
  required to send data to and from the accelerator.

== Design

`MessageData` is a pure abstract base class (held via `std::unique_ptr`)
exposing iteration over `DataSegment`s — contiguous, owning byte buffers.
Backends iterate segments and bulk-transfer each one rather than copying byte-
by-byte.

A `MessageData::Cursor` combines segment iteration with intra-segment offset
into a single, storable bookmark. DMA engines that cannot fit even one segment
save the cursor between buffer-fill calls and resume where they left off.
`Cursor::remaining()` always yields the largest contiguous `span` from the
current position, so every transfer is a bulk copy.

Multiple segments enable zero-copy (de-)serialization of generated types with
variable-length lists (header segment + list-data segment) and scatter-gather
I/O in backends that support it.

`MessageData` becomes abstract, so it cannot be constructed directly. Static
factory functions (`MessageData::create(...)`) return a
`SingleDataSegmentMessageData` to minimize client code churn. Port APIs change
from `MessageData` by-value to `std::unique_ptr<MessageData>`.

A virtual `toFlat()` method moves all segment data into a single
`std::vector<uint8_t>`. The default implementation concatenates segments, but
`SingleDataSegmentMessageData` overrides it to move its internal vector out
with no copy.

Migration: implicit conversions to `vector`/`span` are removed; `getBytes()`,
`getData()`, `takeData()` move to `SingleDataSegmentMessageData`; `as<T>()`
returns by value (copy-out) instead of a pointer.

== API Details

  /// A contiguous, owning byte buffer.
  class DataSegment {
  public:
    virtual ~DataSegment() = default;
    virtual const uint8_t *data() const = 0;
    virtual size_t size() const = 0;
    std::span<const uint8_t> span() const { return {data(), size()}; }
    bool empty() const { return size() == 0; }
  };

  /// DataSegment backed by a std::vector.
  class VectorDataSegment : public DataSegment {
  public:
    VectorDataSegment() = default;
    VectorDataSegment(std::vector<uint8_t> data);
    VectorDataSegment(std::span<const uint8_t> data);
    VectorDataSegment(const uint8_t *data, size_t size);

    const uint8_t *data() const override;
    size_t size() const override;

  private:
    std::vector<uint8_t> storage;
  };

  /// Abstract message: an ordered sequence of DataSegments.
  class MessageData {
  public:
    virtual ~MessageData() = default;

    // --- Segment access ---
    virtual size_t numSegments() const = 0;
    virtual const DataSegment &segment(size_t idx) const = 0;

    // --- Convenience ---
    size_t totalSize() const;
    bool empty() const;
    std::string toHex() const;

    /// Move all data into a flat vector. Consumes the message.
    /// Default: concatenates all segments (copies).
    /// SingleDataSegmentMessageData overrides to move with no copy.
    /// After calling this, the message is empty (totalSize() == 0) and should
    /// not be used.
    virtual std::vector<uint8_t> toFlat();

    // --- Scalar helpers ---
    /// Copy out as T. Throws if totalSize() != sizeof(T).
    template <typename T> T as() const;
    /// Create a single-segment message from a T.
    template <typename T>
    static std::unique_ptr<MessageData> from(T &t);

    // --- Factories (return SingleDataSegmentMessageData) ---
    static std::unique_ptr<MessageData> create();
    static std::unique_ptr<MessageData> create(std::span<const uint8_t> data);
    static std::unique_ptr<MessageData> create(std::vector<uint8_t> data);
    static std::unique_ptr<MessageData> create(const uint8_t *data,
                                               size_t size);

    // --- Cursor (see below) ---
    class Cursor;
    Cursor cursor() const;
  };

  /// Bookmark for incremental consumption of a MessageData.
  /// Tracks position across segment boundaries. Cheaply storable (two
  /// size_t's + a reference).
  class MessageData::Cursor {
  public:
    Cursor(const MessageData &msg);

    /// Contiguous span from current position to end of current segment.
    std::span<const uint8_t> remaining() const;

    /// Advance by `n` bytes, crossing segment boundaries as needed.
    void advance(size_t n);

    /// True when all segments have been consumed.
    bool done() const;

  private:
    const MessageData &msg;
    size_t segIdx = 0;
    size_t offset = 0;
  };

  /// Concrete MessageData owning a single VectorDataSegment.
  class SingleDataSegmentMessageData : public MessageData {
  public:
    SingleDataSegmentMessageData() = default;
    SingleDataSegmentMessageData(std::vector<uint8_t> data);
    SingleDataSegmentMessageData(std::span<const uint8_t> data);
    SingleDataSegmentMessageData(const uint8_t *data, size_t size);

    size_t numSegments() const override;          // returns 1
    const DataSegment &segment(size_t) const override;

    // --- Direct access (single-segment only) ---
    const uint8_t *getBytes() const;
    const std::vector<uint8_t> &getData() const;
    std::vector<uint8_t> takeData();

    /// Override: moves the internal vector out (no copy).
    std::vector<uint8_t> toFlat() override;

  private:
    VectorDataSegment seg;
  };

== Examples

=== Creating messages

  // Simple scalar.
  int32_t value = 42;
  auto msg = MessageData::from(value);

  // From raw bytes.
  uint8_t buf[] = {0xDE, 0xAD, 0xBE, 0xEF};
  auto msg = MessageData::create(buf, sizeof(buf));

  // From a vector (zero-copy move).
  std::vector<uint8_t> vec = buildPayload();
  auto msg = MessageData::create(std::move(vec));

  // Multi-segment (header + list) from generated code.
  class MyStructMessageData : public MessageData {
    VectorDataSegment header;
    VectorDataSegment listData;
  public:
    MyStructMessageData(HeaderT h, std::vector<ItemT> items)
      : header(reinterpret_cast<uint8_t*>(&h), sizeof(h)),
        listData(reinterpret_cast<uint8_t*>(items.data()),
                 items.size() * sizeof(ItemT)) {}
    size_t numSegments() const override { return 2; }
    const DataSegment &segment(size_t i) const override {
      return i == 0 ? static_cast<const DataSegment&>(header)
                    : static_cast<const DataSegment&>(listData);
    }
  };

=== Reading a scalar

  auto msg = port.read();
  int32_t val = msg->as<int32_t>();

=== DMA engine consuming a message

  // Backend with limited buffer space.
  MessageData::Cursor cur = msg->cursor();
  while (!cur.done()) {
    auto chunk = cur.remaining();  // largest contiguous span available
    size_t sent = dma_write(chunk.data(),
                            std::min(chunk.size(), bufferSpace));
    cur.advance(sent);
  }

=== Iterating all segments (scatter-gather)

  for (size_t i = 0; i < msg->numSegments(); ++i) {
    auto sp = msg->segment(i).span();
    sgList.push_back({sp.data(), sp.size()});
  }
