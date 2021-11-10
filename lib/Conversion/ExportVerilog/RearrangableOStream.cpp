//===- RearrangableOStream.cpp - RearrangableOStream implementation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the implementation logic for the RearrangableOStream class.
//
//===----------------------------------------------------------------------===//

#include "RearrangableOStream.h"

using namespace circt;
using namespace ExportVerilog;

void RearrangableOStream::print(raw_ostream &os) {
  for (StringRef segment : getSegments())
    os << segment;
}

void RearrangableOStream::dump() { print(llvm::errs()); }

void RearrangableOStream::Cursor::dump(RearrangableOStream &os) const {
  if (isInvalid()) {
    llvm::errs() << "<invalid cursor>\n";
    return;
  }

  // Determine segment #.
  size_t segmentNo = 0;
  for (auto it = os.segments.begin(), e = os.segments.end(); it != node; ++it) {
    if (it == e) {
      llvm::errs() << "<invalid cursor iterator>, offset=" << offset << '\n';
      return;
    }
    ++segmentNo;
  }
  llvm::errs() << "segment=" << segmentNo << ", offset=" << offset << '\n';
}

/// Split the segment pointed to by the specified cursor, returning a new
/// cursor which is guaranteed to be at the start of the segment (offset=0).
auto RearrangableOStream::splitSegment(Cursor position) -> Cursor {
  assert(!position.isInvalid() && "invalid position");

  // If we're already pointing to the start of a segment, then there is nothing
  // to do.
  if (position.offset == 0)
    return position;

  // The size of the current segment may not be updated if we're looking at the
  // last node we're building into.
  bool isLastSegment = false;
  size_t segmentSize = position.node->size();
  char *segmentPtr = const_cast<char *>(position.node->data());
  if (segmentSize == 0 && position.node == std::prev(segments.end())) {
    segmentSize = remainingChunkPtr - segmentPtr;
    isLastSegment = true;
  }

  assert(position.offset <= segmentSize &&
         "cannot insert into an invalid position");

  // If we're splitting at the end of the segment, we don't need to do anything,
  // but we need to refer to the one after this to get an offset of zero.
  if (position.offset == segmentSize) {
    if (!isLastSegment)
      return Cursor(std::next(position.node), 0);

    // If we were working on the last segment, then close it off and create a
    // new one to continue at the end.
    *position.node = StringRef(segmentPtr, segmentSize);
    segments.push_back(StringRef(remainingChunkPtr, 0));
    return Cursor(std::prev(segments.end()), 0);
  }

  // Keep the first part of the segment at the cursor.
  *position.node = StringRef(segmentPtr, position.offset);

  // Add a new segment after this segment for rest of it.
  segments.insert(std::next(position.node),
                  StringRef(segmentPtr + position.offset,
                            isLastSegment ? 0 : segmentSize - position.offset));
  return Cursor(std::next(position.node), 0);
}

/// Insert the specified string literal text into the buffer at the specified
/// cursor position.  This can split and rearrange segments to make room for
/// it.
void RearrangableOStream::insertLiteral(Cursor position, StringRef what) {
  assert(!position.isInvalid() && "invalid position");
  flush();

  // If there is space in the current chunk and we just need to memcpy a
  // small amount of text, do so to avoid creating new segments.
  if (position.node->empty() && position.node == std::prev(segments.end()) &&
      what.size() <= remainingChunkSize) {
    char *segmentPtr = const_cast<char *>(position.node->data());
    size_t segmentSize = remainingChunkPtr - segmentPtr;

    assert(position.offset <= segmentSize &&
           "cannot insert into an invalid position");

    size_t sizeToCopy = segmentSize - position.offset;
    if (sizeToCopy < 128) {
      // Move over the existing text to make room, then plop "what" in the hole.
      char *cursorPtr = segmentPtr + position.offset;
      memmove(cursorPtr + what.size(), cursorPtr, sizeToCopy);
      memcpy(cursorPtr, what.data(), what.size());
      remainingChunkPtr += what.size();
      remainingChunkSize -= what.size();
      return;
    }
  }

  // Split the segment at the cursor point.
  position = splitSegment(position);

  // Add the string literal in between as a new segment.
  segments.insert(position.node, what);
}

/// Rearrange the block of text in the "[fromBegin, fromEnd)" region to appear
/// before the position cursor.  This returns a new position Cursor reflecting
/// a new Cursor after the inserted text.
RearrangableOStream::Cursor
RearrangableOStream::moveRangeBefore(Cursor position, Cursor fromBegin,
                                     Cursor fromEnd) {
  // Make sure nothing is buffered in raw_ostream, it is all in the segments.
  flush();
  // Split segments at the places we need to move things around.  This ensures
  // that the offsets of each cursor are all zero.
  fromEnd = splitSegment(fromEnd);
  fromBegin = splitSegment(fromBegin);
  position = splitSegment(position);

  // Just move the segments into place now.
  segments.splice(position.node, segments, fromBegin.node, fromEnd.node);
  return position;
}

/// Flushes the stream contents to the target string to the segment list, and
/// returns the segment list for inspection.
const std::list<StringRef> &RearrangableOStream::getSegments() {
  // Make sure nothing is buffered anywhere in the stream.
  flush();
  // Finish off the last segment if neccesary.
  splitCurrentSegment();
  return segments;
}

/// Split and close off the currently emitted segment, and start a new one.
void RearrangableOStream::splitCurrentSegment() {
  auto &curSegment = segments.back();
  assert(curSegment.size() == 0 && "segment shouldn't be finished yet");

  // If nothing is in flight, we are done.
  if (curSegment.data() == remainingChunkPtr)
    return;

  // Set the length of the last segment correctly.
  curSegment =
      StringRef(curSegment.data(), remainingChunkPtr - curSegment.data());
  // Start the next segment.
  segments.push_back(StringRef(remainingChunkPtr, 0));
}

void RearrangableOStream::write_impl(const char *ptr, size_t size) {
  if (size == 0)
    return;

  // If we are out of space, allocate another chunk, which is at least twice
  // as big as the last one.
  if (size > remainingChunkSize) {
    splitCurrentSegment();
    remainingChunkSize = lastChunkSize = std::max(lastChunkSize * 2, size);
    remainingChunkPtr = (char *)malloc(lastChunkSize);
    chunks.push_back(remainingChunkPtr);
    segments.back() = StringRef(remainingChunkPtr, 0);
  }

  // Copy the data in and remember we did.
  memcpy(remainingChunkPtr, ptr, size);
  remainingChunkPtr += size;
  remainingChunkSize -= size;
}