//===- Common.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT (lib/dialect/ESI/runtime/cpp/).
//
//===----------------------------------------------------------------------===//

#include "esi/Common.h"

#include <iostream>
#include <sstream>

using namespace esi;

std::string MessageData::toHex() const {
  std::ostringstream ss;
  ss << std::hex;
  for (size_t i = 0, e = data.size(); i != e; ++i) {
    // Add spaces every 8 bytes.
    if (i % 8 == 0 && i != 0)
      ss << ' ';
    // Add an extra space every 64 bytes.
    if (i % 64 == 0 && i != 0)
      ss << ' ';
    ss << static_cast<unsigned>(data[i]);
  }
  return ss.str();
}

std::string esi::toHex(void *val) {
  return toHex(reinterpret_cast<uint64_t>(val));
}

std::string esi::toHex(uint64_t val) {
  std::ostringstream ss;
  ss << std::hex << val;
  return ss.str();
}

//===----------------------------------------------------------------------===//
// SegmentedMessageData
//===----------------------------------------------------------------------===//

size_t SegmentedMessageData::totalSize() const {
  size_t total = 0;
  for (size_t i = 0, e = numSegments(); i < e; ++i)
    total += segment(i).size;
  return total;
}

bool SegmentedMessageData::empty() const { return totalSize() == 0; }

MessageData SegmentedMessageData::toMessageData() const {
  size_t total = totalSize();
  if (total == 0)
    return MessageData();
  std::vector<uint8_t> buf;
  buf.reserve(total);
  for (size_t i = 0, e = numSegments(); i < e; ++i) {
    Segment seg = segment(i);
    if (seg.size == 0)
      continue;
    buf.insert(buf.end(), seg.data, seg.data + seg.size);
  }
  return MessageData(std::move(buf));
}

//===----------------------------------------------------------------------===//
// SegmentedMessageDataCursor
//===----------------------------------------------------------------------===//

std::span<const uint8_t> SegmentedMessageDataCursor::remaining() const {
  // Skip past any empty segments.
  while (!done()) {
    Segment seg = msg.segment(segIdx);
    if (seg.size > offset)
      return {seg.data + offset, seg.size - offset};
    // This shouldn't happen in normal use (advance handles crossing),
    // but guard against empty segments at the current position.
    break;
  }
  return {};
}

void SegmentedMessageDataCursor::advance(size_t n) {
  while (n > 0 && !done()) {
    Segment seg = msg.segment(segIdx);
    size_t left = seg.size - offset;
    if (left == 0) {
      // Skip empty segments.
      ++segIdx;
      offset = 0;
      continue;
    }
    if (n < left) {
      offset += n;
      return;
    }
    n -= left;
    ++segIdx;
    offset = 0;
  }
}

bool SegmentedMessageDataCursor::done() const {
  return segIdx >= msg.numSegments();
}

void SegmentedMessageDataCursor::reset() {
  segIdx = 0;
  offset = 0;
}
