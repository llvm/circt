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

//===----------------------------------------------------------------------===//
// MessageData
//===----------------------------------------------------------------------===//

size_t MessageData::totalSize() const {
  size_t total = 0;
  for (size_t i = 0, n = numSegments(); i < n; ++i)
    total += segment(i).size();
  return total;
}

bool MessageData::empty() const { return totalSize() == 0; }

std::string MessageData::toHex() const {
  std::ostringstream ss;
  ss << std::hex;
  size_t byteIdx = 0;
  for (size_t s = 0, ns = numSegments(); s < ns; ++s) {
    auto &seg = segment(s);
    for (size_t i = 0, e = seg.size(); i < e; ++i, ++byteIdx) {
      if (byteIdx % 8 == 0 && byteIdx != 0)
        ss << ' ';
      if (byteIdx % 64 == 0 && byteIdx != 0)
        ss << ' ';
      ss << static_cast<unsigned>(seg.data()[i]);
    }
  }
  return ss.str();
}

std::vector<uint8_t> MessageData::toFlat() {
  std::vector<uint8_t> result;
  result.reserve(totalSize());
  for (size_t i = 0, n = numSegments(); i < n; ++i) {
    auto &seg = segment(i);
    result.insert(result.end(), seg.data(), seg.data() + seg.size());
  }
  return result;
}

std::vector<uint8_t> MessageData::toFlatCopy() const {
  std::vector<uint8_t> result;
  result.reserve(totalSize());
  for (size_t i = 0, n = numSegments(); i < n; ++i) {
    auto &seg = segment(i);
    result.insert(result.end(), seg.data(), seg.data() + seg.size());
  }
  return result;
}

MessageData::Cursor MessageData::cursor() const { return Cursor(*this); }

std::unique_ptr<MessageData> MessageData::create() {
  return std::make_unique<SingleDataSegmentMessageData>();
}
std::unique_ptr<MessageData>
MessageData::create(std::span<const uint8_t> data) {
  return std::make_unique<SingleDataSegmentMessageData>(data);
}
std::unique_ptr<MessageData>
MessageData::create(std::vector<uint8_t> data) {
  return std::make_unique<SingleDataSegmentMessageData>(std::move(data));
}
std::unique_ptr<MessageData> MessageData::create(const uint8_t *data,
                                                 size_t size) {
  return std::make_unique<SingleDataSegmentMessageData>(data, size);
}

std::string esi::toHex(void *val) {
  return toHex(reinterpret_cast<uint64_t>(val));
}

std::string esi::toHex(uint64_t val) {
  std::ostringstream ss;
  ss << std::hex << val;
  return ss.str();
}
