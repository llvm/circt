//===- Compression.h - Compression Utilities --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for transparently decompressing input files, supplementing the
// ones from llvm::compression. These are guarded by CIRCT_ENABLE_ZSTD.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_COMPRESSION_H
#define CIRCT_SUPPORT_COMPRESSION_H

#include "circt/Support/LLVM.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>

namespace circt {

/// Returns true if this build of CIRCT was compiled with zstd support
/// (CIRCT_ENABLE_ZSTD).
bool isZstdAvailable();

/// Returns true if the given buffer starts with the zstd magic number, i.e., it
/// looks like a zstd-compressed frame.
bool hasZstdMagic(const llvm::MemoryBuffer &buffer);

/// Selects where the uncompressed output of `decompressZstd` is materialized.
enum class ZstdOutput {
  /// Decompress into a single in-memory buffer. Peak additional memory is ~1x
  /// the uncompressed size (plus the still-mapped compressed input).
  InMemory,
  /// Decompress into a temporary file on disk and return a read-only mmap of
  /// it. The temporary file is removed automatically once the returned buffer
  /// (and thus the mapping) is destroyed. This keeps resident memory bounded
  /// by the OS page cache rather than committing the entire uncompressed size
  /// as anonymous memory, which is preferable for very large inputs.
  TempFile,
};

/// Decompress a zstd-compressed memory buffer and return the uncompressed
/// contents as a new, null-terminated memory buffer. On failure, returns an
/// `llvm::Error` describing the problem.
///
/// When the compressed frame advertises its uncompressed size, the output is
/// allocated exactly once, avoiding the incremental-reallocation and
/// double-buffering overhead of the streaming fallback. `output` selects
/// whether the result is held in memory or spilled to a temporary file and
/// mmap'd; the latter is recommended for very large (multi-GiB) inputs.
///
/// This requires that CIRCT was built with zstd support (CIRCT_ENABLE_ZSTD).
/// If not, this fails with an appropriate error.
llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
decompressZstd(std::unique_ptr<llvm::MemoryBuffer> input,
               ZstdOutput output = ZstdOutput::InMemory);

/// If `input` looks like a zstd-compressed buffer (either because `filename`
/// ends in `.zst` or because the buffer starts with the zstd magic number),
/// transparently decompress it and return the uncompressed buffer. Otherwise
/// the input buffer is returned unchanged.
///
/// On decompression failure, returns an `llvm::Error`. If the buffer looks
/// compressed but CIRCT was built without zstd support, this also fails.
llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
maybeDecompressZstd(std::unique_ptr<llvm::MemoryBuffer> input,
                    StringRef filename,
                    ZstdOutput output = ZstdOutput::InMemory);

} // namespace circt

#endif // CIRCT_SUPPORT_COMPRESSION_H
