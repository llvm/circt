; REQUIRES: zstd
; Verify that firtool transparently decompresses zstd-compressed input files.
; We compress the FIRRTL circuit below into a temporary `.fir.zst` file and
; then run firtool on it, checking that the output matches the uncompressed
; behaviour.

; RUN: rm -rf %t && mkdir -p %t
; RUN: sed -n '/^FIRRTL version/,$p' %s > %t/input.fir
; RUN: zstd -q -f %t/input.fir -o %t/input.fir.zst

; Decompression keyed on the `.fir.zst` suffix (format autodetected as FIR).
; RUN: firtool %t/input.fir.zst | FileCheck %s

; Decompression keyed on the zstd magic number, with an explicit format and a
; non-suffixed filename.
; RUN: cp %t/input.fir.zst %t/input.bin
; RUN: firtool -format=fir %t/input.bin | FileCheck %s

; Same, but decompressing via a temporary file + mmap instead of in memory.
; RUN: firtool --zstd-decompress-to-temp-file %t/input.fir.zst | FileCheck %s

; A frame without an embedded content size exercises the streaming fallback.
; RUN: zstd -q -c --no-content-size %t/input.fir > %t/input-nosize.fir.zst
; RUN: firtool %t/input-nosize.fir.zst | FileCheck %s
; RUN: firtool --zstd-decompress-to-temp-file %t/input-nosize.fir.zst | FileCheck %s

; CHECK: module Foo(
; CHECK: assign b = a;
FIRRTL version 4.0.0
circuit Foo:
  public module Foo:
    input a : UInt<8>
    output b : UInt<8>
    connect b, a
