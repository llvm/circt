# ESI Accelerator Runtime changelog

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
