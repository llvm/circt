//===- RearrangableOStream.h - Helper for ExportVerilog ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_EXPORTVERILOG_REARRANGABLEOSTREAM_H
#define CONVERSION_EXPORTVERILOG_REARRANGABLEOSTREAM_H

#include "circt/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <list>

namespace circt {
namespace ExportVerilog {

/// This class is a raw_ostream that supports streaming a wide variety of text
/// data into it, but explicitly manages the orders of chunks.  It is used by
/// ExportVerilog to support efficient insertion of text into text regions that
/// have already been generated.  There are thre major concepts here: chunks,
/// segments, and cursors.
///
/// A "chunk" is a slab of memory that we throw text into.  We use a simple
/// size-doubling policy for memory allocation and just insert into the back of
/// the chunk.  This is what makes this a fancy raw_ostream.
///
/// "Segments" are slices of chunks represented as StringRef's.  These segments
/// are stored in a list and can be reordered to move text around without the
/// generated output.  This is what makes this "Rearrangable".
///
/// "Cursors" are like iterators into the segment stream, which is the unit of
/// reordering.
class RearrangableOStream : public raw_ostream {
public:
  explicit RearrangableOStream() {
    lastChunkSize = 128; // First allocation is 256 bytes.
    remainingChunkPtr = nullptr;
    remainingChunkSize = 0;
    // We always have the unfinished segment on the end of the list.
    segments.push_back(StringRef(nullptr, 0));
  }
  ~RearrangableOStream() override {
    // Free all the data chunks allocated.
    for (char *ptr : chunks)
      free(ptr);
  }

  /// A Cursor represents a position in the ostream.
  struct Cursor {
    Cursor() : node(std::list<StringRef>::iterator()), offset(~size_t()) {}

    bool isInvalid() const { return offset == ~size_t(); }

    void dump(RearrangableOStream &os) const;

  private:
    Cursor(std::list<StringRef>::iterator node, size_t offset)
        : node(node), offset(offset) {}
    std::list<StringRef>::iterator node;
    size_t offset;
    friend class RearrangableOStream;
  };

  /// Return a cursor for the current insertion point.
  Cursor getCursor() {
    // Make sure nothing is buffered anywhere in the stream.
    flush();
    return Cursor(std::prev(segments.end()),
                  size_t(remainingChunkPtr - segments.back().data()));
  }

  /// Rearrange the block of text in the "[fromBegin, fromEnd)" region to appear
  /// before the position cursor.  This returns a new position Cursor reflecting
  /// a new Cursor after the inserted text.
  Cursor moveRangeBefore(Cursor position, Cursor fromBegin, Cursor fromEnd);

  /// Insert the specified string literal text into the buffer at the specified
  /// cursor position.  This can split and rearrange segments to make room for
  /// it.
  void insertLiteral(Cursor position, StringRef what);

  /// Flushes the stream contents to the target string to the segment list, and
  /// returns the segment list for inspection.
  const std::list<StringRef> &getSegments();

  void print(raw_ostream &os);
  void dump();

  /// Split and close off the currently emitted segment, and start a new one.
  /// This can help with cursor invalidation problems.
  void splitCurrentSegment();

private:
  RearrangableOStream(const RearrangableOStream &) = delete;
  void operator=(const RearrangableOStream &) = delete;

  /// Split the segment pointed to by the specified cursor, returning a new
  /// cursor which is guaranteed to be at the start of the segment (offset=0).
  Cursor splitSegment(Cursor position);

  void write_impl(const char *ptr, size_t size) override;

  uint64_t current_pos() const override {
    llvm_unreachable("current_pos not supported");
  }

  /// This keeps track of the allocated blocks of memory so we can deallocate
  /// the memory when done.
  SmallVector<char *> chunks;
  /// This is the total size of the most recent chunk.
  size_t lastChunkSize;
  /// This is the remaining space in the current chunk;
  char *remainingChunkPtr;
  size_t remainingChunkSize;

  /// This keeps track of the output segments being built up.  Each segment
  /// refers to a contiguous piece of text in chunks or might refer to static
  /// data known to outlive the ostream (e.g. a string literal).  The back() of
  /// the list contains the current segment we are building.
  ///
  /// This is a std::list because we need to be able to splice segments of the
  /// list around efficiently, and want stable node iterators.
  std::list<StringRef> segments;
};

} // namespace ExportVerilog
} // namespace circt

#endif // CONVERSION_EXPORTVERILOG_REARRANGABLEOSTREAM_H
