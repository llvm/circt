//===- Common.h - Commonly used classes w/o dependencies --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_COMMON_H
#define ESI_COMMON_H

#include <any>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace esi {
class Type;

//===----------------------------------------------------------------------===//
// Common accelerator description types.
//===----------------------------------------------------------------------===//

struct AppID {
  std::string name;
  std::optional<uint32_t> idx;

  AppID(const std::string &name, std::optional<uint32_t> idx = std::nullopt)
      : name(name), idx(idx) {}

  bool operator==(const AppID &other) const {
    return name == other.name && idx == other.idx;
  }
  bool operator!=(const AppID &other) const { return !(*this == other); }
  friend std::ostream &operator<<(std::ostream &os, const AppID &id);

  std::string toString() const {
    if (idx.has_value())
      return name + "[" + std::to_string(idx.value()) + "]";
    return name;
  }
};
bool operator<(const AppID &a, const AppID &b);

class AppIDPath : public std::vector<AppID> {
public:
  using std::vector<AppID>::vector;

  AppIDPath operator+(const AppIDPath &b) const;
  AppIDPath parent() const;
  std::string toStr() const;
  friend std::ostream &operator<<(std::ostream &os, const AppIDPath &path);
};
bool operator<(const AppIDPath &a, const AppIDPath &b);

struct Constant {
  std::any value;
  std::optional<const Type *> type;
};

struct ModuleInfo {
  std::optional<std::string> name;
  std::optional<std::string> summary;
  std::optional<std::string> version;
  std::optional<std::string> repo;
  std::optional<std::string> commitHash;
  std::map<std::string, Constant> constants;
  std::map<std::string, std::any> extra;
};

/// A description of a service port. Used pretty exclusively in setting up the
/// design.
struct ServicePortDesc {
  std::string name;
  std::string portName;
};

/// Details about how to connect to a particular channel.
struct ChannelAssignment {
  /// The name of the type of connection. Typically, the name of the DMA engine
  /// or "cosim" if a cosimulation channel is being used.
  std::string type;
  /// Implementation-specific options.
  std::map<std::string, std::any> implOptions;
};
using ChannelAssignments = std::map<std::string, ChannelAssignment>;

/// A description of a hardware client. Used pretty exclusively in setting up
/// the design.
struct HWClientDetail {
  AppIDPath relPath;
  ServicePortDesc port;
  ChannelAssignments channelAssignments;
  std::map<std::string, std::any> implOptions;
};
using HWClientDetails = std::vector<HWClientDetail>;
using ServiceImplDetails = std::map<std::string, std::any>;

/// A contiguous, owning byte buffer.
class DataSegment {
public:
  virtual ~DataSegment() = default;
  virtual const uint8_t *data() const = 0;
  virtual size_t size() const = 0;
  std::span<const uint8_t> span() const { return {data(), size()}; }
  bool empty() const { return size() == 0; }
};

/// DataSegment backed by a std::vector.
class VectorDataSegment : public DataSegment {
public:
  VectorDataSegment() = default;
  VectorDataSegment(std::vector<uint8_t> data) : storage(std::move(data)) {}
  VectorDataSegment(std::span<const uint8_t> data)
      : storage(data.begin(), data.end()) {}
  VectorDataSegment(const uint8_t *data, size_t size)
      : storage(data, data + size) {}

  const uint8_t *data() const override { return storage.data(); }
  size_t size() const override { return storage.size(); }

  /// Get the underlying vector (single-segment convenience).
  const std::vector<uint8_t> &getData() const { return storage; }
  /// Move the underlying vector out.
  std::vector<uint8_t> takeData() { return std::move(storage); }

private:
  std::vector<uint8_t> storage;
};

/// Abstract message: an ordered sequence of DataSegments.
class MessageData {
public:
  virtual ~MessageData() = default;

  // --- Segment access ---
  virtual size_t numSegments() const = 0;
  virtual const DataSegment &segment(size_t idx) const = 0;

  // --- Convenience ---
  size_t totalSize() const;
  bool empty() const;
  std::string toHex() const;

  /// Move all data into a flat vector. Destructive — the message should not
  /// be used after calling this. SingleDataSegmentMessageData overrides to
  /// move with no copy.
  virtual std::vector<uint8_t> toFlat();

  /// Copy all data into a flat vector without modifying the message.
  std::vector<uint8_t> toFlatCopy() const;

  // --- Scalar helpers ---
  /// Copy out as T. Throws if totalSize() != sizeof(T).
  template <typename T>
  T as() const {
    if (totalSize() != sizeof(T))
      throw std::runtime_error("Data size does not match type size. Size is " +
                               std::to_string(totalSize()) + ", expected " +
                               std::to_string(sizeof(T)) + ".");
    T val;
    // Gather segments into val.
    uint8_t *dst = reinterpret_cast<uint8_t *>(&val);
    for (size_t i = 0, n = numSegments(); i < n; ++i) {
      auto &seg = segment(i);
      std::memcpy(dst, seg.data(), seg.size());
      dst += seg.size();
    }
    return val;
  }

  /// Create a single-segment message from a T.
  template <typename T>
  static std::unique_ptr<MessageData> from(T &t) {
    return create(reinterpret_cast<const uint8_t *>(&t), sizeof(T));
  }

  // --- Factories (return SingleDataSegmentMessageData) ---
  static std::unique_ptr<MessageData> create();
  static std::unique_ptr<MessageData> create(std::span<const uint8_t> data);
  static std::unique_ptr<MessageData> create(std::vector<uint8_t> data);
  static std::unique_ptr<MessageData> create(const uint8_t *data, size_t size);

  // --- Cursor ---
  class Cursor;
  Cursor cursor() const;
};

/// Bookmark for incremental consumption of a MessageData.
class MessageData::Cursor {
public:
  Cursor(const MessageData &msg) : msg(msg) {}

  /// Contiguous span from current position to end of current segment.
  std::span<const uint8_t> remaining() const {
    auto &seg = msg.segment(segIdx);
    return {seg.data() + offset, seg.size() - offset};
  }

  /// Advance by n bytes, crossing segment boundaries as needed.
  void advance(size_t n) {
    while (n > 0 && !done()) {
      size_t left = msg.segment(segIdx).size() - offset;
      if (n < left) {
        offset += n;
        return;
      }
      n -= left;
      ++segIdx;
      offset = 0;
    }
  }

  /// True when all segments have been consumed.
  bool done() const { return segIdx >= msg.numSegments(); }

private:
  const MessageData &msg;
  size_t segIdx = 0;
  size_t offset = 0;
};

/// Concrete MessageData owning a single VectorDataSegment.
class SingleDataSegmentMessageData : public MessageData {
public:
  SingleDataSegmentMessageData() = default;
  SingleDataSegmentMessageData(std::vector<uint8_t> data) : seg(std::move(data)) {}
  SingleDataSegmentMessageData(std::span<const uint8_t> data) : seg(data) {}
  SingleDataSegmentMessageData(const uint8_t *data, size_t size)
      : seg(data, size) {}

  size_t numSegments() const override { return seg.empty() ? 0 : 1; }
  const DataSegment &segment(size_t) const override { return seg; }

  // --- Direct access (single-segment only) ---
  const uint8_t *getBytes() const { return seg.data(); }
  const std::vector<uint8_t> &getData() const { return seg.getData(); }
  std::vector<uint8_t> takeData() { return seg.takeData(); }

  /// Override: moves the internal vector out (no copy).
  std::vector<uint8_t> toFlat() override { return seg.takeData(); }

  /// Access underlying segment.
  VectorDataSegment &getSegment() { return seg; }
  const VectorDataSegment &getSegment() const { return seg; }

private:
  VectorDataSegment seg;
};

} // namespace esi

std::ostream &operator<<(std::ostream &, const esi::ModuleInfo &);

//===----------------------------------------------------------------------===//
// Functions which should be in the standard library.
//===----------------------------------------------------------------------===//

namespace esi {
std::string toHex(void *val);
std::string toHex(uint64_t val);
} // namespace esi

#endif // ESI_COMMON_H
