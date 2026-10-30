# ESI Accelerator Runtime changelog

## Unreleased

### Breaking changes

- **`connect()` can report when the accelerator end is connected.** Every
  channel port `connect()` overload (`ChannelPort`, `ReadChannelPort`,
  `WriteChannelPort`, the typed ports) and every service port `connect()`
  (`FuncService::Function`, `CallService::Callback`,
  `ChannelService::ToHost`/`FromHost`, `TelemetryService::Metric`, and their
  typed wrappers) takes an optional trailing `ChannelPort::ConnectCallback
  onConnected`. It is called once the accelerator end of the channel(s) is
  connected too, or with an exception if it never will be. `connect()` must
  not block waiting for that. Existing callers are unaffected; backends which
  override `connect()` or `connectImpl()` must add the parameter, and
  `connectImpl()` must arrange for `onConnected` to be called (all in-tree
  backends currently call it immediately). `utils::fanInCallbacks()` combines
  the callbacks of several channels. In Python, service ports' `connect()`
  returns a `ConnectFuture`; channel `Port.connect()` still returns the port
  (for chaining) and stores it in `Port.connected_future`.
- **`HostMemRegion` is now defined at namespace scope** as
  `esi::services::HostMemRegion`. `services::HostMem::HostMemRegion` remains
  as an alias, so existing source compiles unchanged, but code linking against
  the runtime must be rebuilt.
- **`HostMem::allocate()` can fail and return nullptr** (`None` in Python),
  and always does so when `size` is 0. Callers must check the result.
  `HostMem` implementations must follow the same contract; the built-in
  backends now return nullptr rather than crashing or throwing when an
  allocation fails.
- **`SegmentedMessageDataCursor::reset()` has been removed.** Rewinding to
  retransmit is unsafe once segments may have been `take()`n. To start over,
  construct a new cursor (e.g. via `std::optional::emplace()`).

### Added

- **Non-blocking segmented writes.** `WriteChannelPort` gains a virtual
  `bool tryWrite(std::unique_ptr<SegmentedMessageData> &msg)` and
  `TypedWritePort<T>` gains `bool tryWrite(std::unique_ptr<T> &data)`. They
  take ownership of the message (leaving the pointer null) only when they
  return true; on false the caller keeps the message and can retry. The
  default implementation flattens the message and calls the existing
  `tryWrite(const MessageData &)`; backends may override it for
  scatter-gather support.
- **Multiple outstanding host memory reads per client.**
  `HostMemReadReqSplitter` takes a new optional `max_outstanding` parameter
  (default `1`, which keeps the previous one-request-at-a-time behavior). With
  a larger value it accepts up to that many logical read requests while
  earlier ones are still returning data, and frames each request's response
  burst (`last`, `valid_bytes`) from a small metadata FIFO. Single-chunk
  requests can be accepted every cycle and response bursts flow back to back
  with no bubble. The upstream must return each client's response words in
  request order. `HostmemReadProcessor` and `ChannelHostMem` pass the limit
  through as `max_outstanding_reads` (default `1`). `CosimBSP` and
  `CosimBSP_DMA` take the same parameter and default it to `4`, since the
  cosim host answers reads in order.
- `HostMem.start()` is now available from Python, so Python code can enable
  the host memory service (needed before the accelerator reads host memory
  under cosim).
- `CosimBSP` accepts a `channel_service` implementation class for custom channel
  transports, including engines shared by multiple channels. The BSP satisfies
  the implementation's MMIO and HostMem requests. This option is mutually
  exclusive with the existing `dma_engine_pair` shorthand.
- `ChannelEngineService` connects an engine's `hostmem_read_list` port to
  HostMem as a burst (`read_list`) read, as it already does for `hostmem_read`
  and `hostmem_write`. Engines can now read many host memory words per request.
- **Zero-copy DMA from HostMem regions through segmented messages.** A
  `Segment` can point (non-owningly) at the `HostMemRegion` holding its bytes
  via the new `region` field, and `Segment::getDeviceAddress()` returns the
  device address of those bytes, so a scatter-gather backend can DMA
  region-backed segments directly. The message keeps its regions alive.
  Optionally, a message which owns its
  regions exclusively can override the new virtual
  `SegmentedMessageData::take(segIdx)` to transfer a segment's region to the
  backend once transmitted (to keep, re-use, or pool); by default it returns
  nullptr. Once taken, a segment is no longer accessible via `segment()`.
  Several segments may share a region; `take()` returns it once the last of
  them has been taken. `SegmentedMessageDataCursor::remainingSegment()` returns
  the unconsumed part of the current segment with its `region`.
  `Segment{ptr, size}` initialization is unchanged. Also added
  `HostMemRegion::getDeviceAddress(ptr, size)`, a bounds-checked device
  address for a host range within the region. See `docs/MessageData.md`.
- **`services::HostMemAllocator` interface.** Anything which can allocate
  `HostMemRegion`s; `HostMem` now implements it. `HostMem::Options` is now
  defined on the interface (still accessible as `HostMem::Options`). Code
  which only needs to allocate regions (e.g. message types which keep their
  data in HostMem) can accept a `HostMemAllocator` so callers can supply
  other allocators, such as pools.

## 0.8.0

### Breaking changes

- **Union field alignment has changed.** Union fields narrower than the union
  now align at the least-significant bit (LSB), rather than at the
  most-significant bit (MSB). Unused bits are padded with zeros on the MSB side.
  This matches CIRCT's corrected `hw.union` bitcast layout and PyCDE's
  `UnionType` layout
  ([#11226](https://github.com/llvm/circt/pull/11226)).

  `esiaccel` 0.8.0 requires accelerator images built with PyCDE 0.12.1. Images
  built with earlier PyCDE versions use the previous union layout and are not
  wire-compatible for affected fields. Regenerate accelerator images and
  update the runtime together.
