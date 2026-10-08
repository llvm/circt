# ESI Accelerator Runtime changelog

## Unreleased

### Breaking changes

- **`HostMemRegion` is now defined at namespace scope** as
  `esi::services::HostMemRegion`. `services::HostMem::HostMemRegion` remains
  as an alias, so existing source compiles unchanged, but code linking against
  the runtime must be rebuilt.
- **`HostMem::allocate()` can fail and return nullptr** (`None` in Python),
  and always does so when `size` is 0. Callers must check the result.
  `HostMem` implementations must follow the same contract; the built-in
  backends now return nullptr rather than crashing or throwing when an
  allocation fails.

### Added

- `CosimBSP` accepts a `channel_service` implementation class for custom channel
  transports, including engines shared by multiple channels. The BSP satisfies
  the implementation's MMIO and HostMem requests. This option is mutually
  exclusive with the existing `dma_engine_pair` shorthand.
- **Zero-copy transfer of HostMem regions through segmented messages.** A
  `Segment` can point (non-owningly) at the `HostMemRegion` holding its bytes
  via the new `region` field, and `Segment::getDeviceAddress()` returns the
  device address of those bytes. The new virtual
  `SegmentedMessageData::take(segIdx)` transfers ownership of a segment's
  region from the message to the caller (by default, it returns nullptr), so a
  scatter-gather backend can DMA region-backed segments directly and then keep,
  re-use, or pool the regions. Once taken, a segment is no longer accessible
  via `segment()`. Several segments may share a region; `take()` returns it
  once the last of them has been taken. `Segment{ptr, size}` initialization
  is unchanged. Also
  added `HostMemRegion::getDeviceAddress(ptr, size)`, a bounds-checked device
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
