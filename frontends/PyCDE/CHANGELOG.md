# PyCDE changelog

## Unreleased

### Changes

- **`seq.FIFO` status flags are registers.** The `seq.fifo` lowering now keeps
  `full`, `empty` and the almost-full/almost-empty flags in registers, and no
  longer compares the occupancy count in front of the pointer and count
  updates. This shortens the logic in and around every FIFO
  ([#11321](https://github.com/llvm/circt/issues/11321)).

  Reading an empty FIFO or writing a full one was, and remains, a protocol
  violation, caught by simulation assertions. The lowering used to partly guard
  against such accesses; it no longer does, so they now corrupt the FIFO.
  Designs that never make them behave exactly as before, cycle for cycle. The
  assertions no longer fire while the FIFO is in reset.

- **`seq.FIFO` read latency is on the read data.** With `rd_latency` 2 or
  more, the memory is read in the cycle the entry is popped and the rest of the
  latency is registers on the read data, rather than on the read address. On
  FPGAs this lets the second register be the RAM block's output register. A
  popped slot is now free at once, so `full` and `almost_full` no longer stay
  asserted while a pop's read data is in flight, as they did in 0.13.0.

## 0.13.0

### Breaking changes

- **The ESI channel ASM format has changed.** Channel types now require an
  explicit signaling protocol, for example `!esi.channel<i8, ValidReady>`
  instead of `!esi.channel<i8>`. Older ASM exports containing `esi.channel`
  types without this protocol cannot be imported by PyCDE 0.13.0.

  **Migration:** Regenerate ASM exports with PyCDE 0.13.0 or newer before
  importing them. When migrating existing ASM manually, add `, ValidReady`
  to channel types that previously used the implicit valid/ready protocol.

### Fixed

- **`esi.TaggedDemux` duplicated messages under backpressure.** Its input was
  ready only when *every* output was ready, yet each output saw `valid`
  whenever the input's tag selected it. So if the selected output was ready
  but another wasn't, the selected output received the same message every
  cycle. The input is now consumed exactly when the selected output accepts it
  (messages whose tag selects no output are dropped, as before).

## 0.12.1

### Breaking changes

- **ESI manifests now report ABI version 1.** The manifest's `apiVersion`
  field changes from 0 to 1, with matching updates to the MMIO and cosimulation
  version metadata. This marks the compatibility break caused by the CIRCT
  union-layout bug fix included in PyCDE 0.12.0; it does not introduce another
  layout change ([#11225](https://github.com/llvm/circt/pull/11225)).

  **Migration:** Regenerate hardware with PyCDE 0.12.1 or newer when using an
  ESI runtime that requires ABI version 1. PyCDE 0.12.0 has the corrected union
  layout but still emits version-0 manifests, which that runtime rejects.
  Update the compiler and runtime together rather than editing manifest
  version fields by hand.

## 0.12.0

### Breaking changes

- **`UnionType` field alignment has changed.** In generated SystemVerilog,
  fields narrower than the union now align at the least-significant bit (LSB)
  by default, rather than at the most-significant bit (MSB). Unused bits are
  padded with zeros on the MSB side when constructing a union value. This
  corrects the previous Verilog export behavior to match CIRCT's documented
  union bitcast layout ([#11135](https://github.com/llvm/circt/pull/11135)).

  For example, in `UnionType([("wide", Bits(32)), ("narrow", Bits(16))])`,
  `narrow` now occupies bits `[15:0]` instead of `[31:16]`. Constructing the
  union with `("narrow", 0x1234)` now produces the 32-bit value `0x00001234`,
  rather than `0x12340000`.

  An explicit field offset (the third element of a field tuple) is now measured
  from the LSB: a field of width `w` at offset `o` occupies bits
  `[o + w - 1:o]`. Union construction zero-pads below the field by `o` bits
  and above it to fill the union width.

  **Migration:** Review bitcasts, manual bit slicing, serialized data, and
  external hardware or software interfaces that depend on union layout.
  Regenerate hardware and update matching consumers together; old and new
  layouts are not wire-compatible for affected fields. For hardware-only
  unions that intentionally require MSB alignment, specify an explicit offset
  of `union_width - field_width` for each field. The ESI runtime (version 0.8.0)
  has also been updated to match the corrected alignment
  ([#11226](https://github.com/llvm/circt/pull/11226)); ESI does not support
  nonzero union field offsets.
