//===- Compression.cpp - Compression Utilities ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for transparently decompressing input files.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/Compression.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#ifdef CIRCT_ENABLE_ZSTD
#include <zstd.h>
#endif

using namespace circt;

/// The zstd magic number that prefixes every zstd frame, in little-endian byte
/// order (0xFD2FB528).
static constexpr char kZstdMagic[4] = {'\x28', '\xB5', '\x2F', '\xFD'};

bool circt::isZstdAvailable() {
#ifdef CIRCT_ENABLE_ZSTD
  return true;
#else
  return false;
#endif
}

bool circt::hasZstdMagic(const llvm::MemoryBuffer &buffer) {
  StringRef data = buffer.getBuffer();
  auto size = sizeof(kZstdMagic);
  return data.size() >= size &&
         data.take_front(size) == StringRef(kZstdMagic, size);
}

#ifdef CIRCT_ENABLE_ZSTD
namespace {
/// RAII wrapper for a ZSTD_DCtx.
struct ZstdDCtx {
  ZSTD_DCtx *ctx = ZSTD_createDCtx();
  ~ZstdDCtx() {
    if (ctx)
      ZSTD_freeDCtx(ctx);
  }
  explicit operator bool() const { return ctx != nullptr; }
};

/// A MemoryBuffer that maps a temporary file read-only and removes that file
/// once the mapping is destroyed. This lets the OS page cache back very large
/// decompressed inputs instead of committing them as anonymous memory.
class TempFileMemoryBuffer : public llvm::MemoryBuffer {
public:
  TempFileMemoryBuffer(std::unique_ptr<llvm::MemoryBuffer> mapping,
                       std::string path, StringRef name)
      : mapping(std::move(mapping)), path(std::move(path)), name(name.str()) {
    init(this->mapping->getBufferStart(), this->mapping->getBufferEnd(),
         /*RequiresNullTerminator=*/true);
  }

  ~TempFileMemoryBuffer() override {
    // Drop the mapping first, then remove the backing file.
    mapping.reset();
    llvm::sys::fs::remove(path);
  }

  StringRef getBufferIdentifier() const override { return name; }
  BufferKind getBufferKind() const override { return mapping->getBufferKind(); }

private:
  std::unique_ptr<llvm::MemoryBuffer> mapping;
  std::string path;
  std::string name;
};

/// Report the uncompressed size advertised by the first frame, or 0 if it is
/// unknown or the header is too small/invalid to tell.
static uint64_t frameContentSizeHint(StringRef data) {
  unsigned long long size = ZSTD_getFrameContentSize(data.data(), data.size());
  if (size == ZSTD_CONTENTSIZE_UNKNOWN || size == ZSTD_CONTENTSIZE_ERROR)
    return 0;
  return size;
}

/// Returns true if `data` consists of exactly one zstd frame. Only then does
/// the first frame's advertised content size describe the whole output, making
/// it safe to preallocate a single exact-size buffer.
/// `ZSTD_getFrameContentSize` only ever reports the first frame, so for
/// multi-frame inputs we must fall back to the streaming path to avoid silently
/// dropping trailing frames.
static bool isSingleZstdFrame(StringRef data) {
  size_t frameSize = ZSTD_findFrameCompressedSize(data.data(), data.size());
  return !ZSTD_isError(frameSize) && frameSize == data.size();
}

/// Run the zstd streaming decompressor over `data`, invoking `sink` with each
/// produced chunk. Returns an error on malformed or truncated input.
template <typename SinkFn>
static llvm::Error streamDecompress(ZSTD_DCtx *dctx, StringRef data,
                                    SinkFn sink) {
  ZSTD_inBuffer in = {data.data(), data.size(), 0};
  llvm::SmallVector<char, 0> scratch;
  const size_t chunkSize = ZSTD_DStreamOutSize();
  scratch.resize_for_overwrite(chunkSize);

  size_t lastRet = 0;
  bool sawFrameEnd = false;
  while (in.pos < in.size) {
    ZSTD_outBuffer out = {scratch.data(), chunkSize, 0};
    lastRet = ZSTD_decompressStream(dctx, &out, &in);
    if (ZSTD_isError(lastRet))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "failed to decompress zstd input: " +
              std::string(ZSTD_getErrorName(lastRet)));
    if (auto err = sink(StringRef(scratch.data(), out.pos)))
      return err;
    sawFrameEnd = (lastRet == 0);
  }

  // A non-zero final return value means the input ended mid-frame, i.e. it was
  // truncated.
  if (!sawFrameEnd && lastRet != 0)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "failed to decompress zstd input: unexpected end of input");
  return llvm::Error::success();
}
} // namespace
#endif // CIRCT_ENABLE_ZSTD

#ifdef CIRCT_ENABLE_ZSTD
/// Decompress into a single in-memory buffer. If the frame advertises its size
/// we allocate exactly once (and decompress straight into it, avoiding both the
/// incremental SmallVector reallocations and the final copy). Otherwise we fall
/// back to growing a buffer and copying once at the end.
static llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
decompressZstdInMemory(ZSTD_DCtx *dctx, StringRef data, StringRef name) {
  uint64_t hint = frameContentSizeHint(data);
  if (hint != 0 && isSingleZstdFrame(data)) {
    // Fast path: allocate the exact output size once. getNewUninitMemBuffer
    // reserves an extra byte and null-terminates it for us.
    auto buffer = llvm::WritableMemoryBuffer::getNewUninitMemBuffer(hint, name);
    if (!buffer)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "failed to allocate " + llvm::Twine(hint) +
                                         " bytes for decompressed zstd input");

    char *dst = buffer->getBufferStart();
    size_t written = 0;
    ZSTD_inBuffer in = {data.data(), data.size(), 0};
    size_t lastRet = 0;
    bool sawFrameEnd = false;
    while (in.pos < in.size && written < hint) {
      ZSTD_outBuffer out = {dst + written, hint - written, 0};
      lastRet = ZSTD_decompressStream(dctx, &out, &in);
      if (ZSTD_isError(lastRet))
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "failed to decompress zstd input: " +
                std::string(ZSTD_getErrorName(lastRet)));
      written += out.pos;
      sawFrameEnd = (lastRet == 0);
    }

    // The advertised size only covers the first frame; if it does not match
    // what we produced (e.g. truncated, or multiple concatenated frames), fall
    // through to an error rather than returning a partial buffer.
    if (written != hint)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "zstd input does not match its advertised uncompressed size");
    if (!sawFrameEnd && lastRet != 0)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "failed to decompress zstd input: unexpected end of input");
    return std::unique_ptr<llvm::MemoryBuffer>(std::move(buffer));
  }

  // Fallback path: size unknown, grow-and-copy.
  llvm::SmallVector<char, 0> output;
  if (auto err = streamDecompress(dctx, data, [&](StringRef chunk) {
        output.append(chunk.begin(), chunk.end());
        return llvm::Error::success();
      }))
    return std::move(err);
  return llvm::MemoryBuffer::getMemBufferCopy(
      StringRef(output.data(), output.size()), name);
}

/// Decompress into a temporary file and return a read-only mmap of it, wrapped
/// so the file is removed when the buffer is destroyed. Keeps resident memory
/// bounded by the page cache rather than the full uncompressed size.
static llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
decompressZstdToTempFile(ZSTD_DCtx *dctx, StringRef data, StringRef name) {
  auto temp = llvm::sys::fs::TempFile::create("circt-zstd-%%%%%%.fir");
  if (!temp)
    return temp.takeError();

  {
    llvm::raw_fd_ostream os(temp->FD, /*shouldClose=*/false);
    llvm::Error err =
        streamDecompress(dctx, data, [&](StringRef chunk) -> llvm::Error {
          os.write(chunk.data(), chunk.size());
          if (os.has_error())
            return llvm::createStringError(
                os.error(), "failed to write decompressed output");
          return llvm::Error::success();
        });
    os.flush();
    if (err) {
      llvm::consumeError(temp->discard());
      return std::move(err);
    }
    if (os.has_error()) {
      llvm::consumeError(temp->discard());
      return llvm::createStringError(os.error(),
                                     "failed to write decompressed output");
    }
  }

  std::string path = temp->TmpName;
  // Keep the file on disk; ownership transfers to TempFileMemoryBuffer, which
  // removes it once the mapping is dropped.
  if (auto err = temp->keep())
    return std::move(err);

  auto mapping = llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                             /*RequiresNullTerminator=*/true);
  if (!mapping) {
    llvm::sys::fs::remove(path);
    return llvm::createStringError(mapping.getError(),
                                   "failed to map decompressed temporary file");
  }

  return std::unique_ptr<llvm::MemoryBuffer>(
      std::make_unique<TempFileMemoryBuffer>(std::move(*mapping),
                                             std::move(path), name));
}
#endif // CIRCT_ENABLE_ZSTD

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
circt::decompressZstd(std::unique_ptr<llvm::MemoryBuffer> input,
                      ZstdOutput output) {
#ifndef CIRCT_ENABLE_ZSTD
  (void)input;
  (void)output;
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "cannot decompress zstd input: CIRCT was built without zstd "
      "support (CIRCT_ENABLE_ZSTD)");
#else
  ZstdDCtx dctx;
  if (!dctx)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to create zstd decompression "
                                   "context");

  StringRef data = input->getBuffer();
  StringRef name = input->getBufferIdentifier();
  switch (output) {
  case ZstdOutput::InMemory:
    return decompressZstdInMemory(dctx.ctx, data, name);
  case ZstdOutput::TempFile:
    return decompressZstdToTempFile(dctx.ctx, data, name);
  }
  llvm_unreachable("unhandled ZstdOutput");
#endif
}

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
circt::maybeDecompressZstd(std::unique_ptr<llvm::MemoryBuffer> input,
                           StringRef filename, ZstdOutput output) {
  if (!input)
    return std::move(input);

  bool looksCompressed = filename.ends_with(".zst") || hasZstdMagic(*input);
  if (!looksCompressed)
    return std::move(input);

  return decompressZstd(std::move(input), output);
}
