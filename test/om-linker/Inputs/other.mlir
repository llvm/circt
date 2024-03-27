// This module contains operations from other dialects which are expected to
// work with the om-linker tool.  The contents of this file does not matter,
// just that one op from each dependent dialect is added.
module {
  hw.module @HW(in %a: i1, out b: i1) {
    %x = sv.constantX : i1
    %0 = comb.xor %a, %x : i1
    hw.output %0 : i1
  }
  emit.file "foo.sv" sym @Emit {}
}
