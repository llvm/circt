//===- ESIRuntimeSegmentTest.cpp - Segment HostMem region tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "esi/Common.h"
#include "esi/Services.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace esi;
using services::HostMem;
using services::HostMemRegion;

namespace {

static_assert(std::is_same_v<HostMem::HostMemRegion, HostMemRegion>);
static_assert(std::is_base_of_v<services::HostMemAllocator, HostMem>);
static_assert(std::is_aggregate_v<Segment>);

/// A region whose host and device addresses are arbitrary integers. Never
/// dereferenced.
struct FakeRegion : public HostMemRegion {
  FakeRegion(uintptr_t host, uintptr_t dev, std::size_t size)
      : host(host), dev(dev), size(size) {}
  void *getPtr() const override { return reinterpret_cast<void *>(host); }
  void *getDevicePtr() const override { return reinterpret_cast<void *>(dev); }
  std::size_t getSize() const override { return size; }
  uintptr_t host, dev;
  std::size_t size;
};

/// A region backed by a real buffer. Sets `*destroyed` when deconstructed.
struct BufferRegion : public HostMemRegion {
  BufferRegion(std::size_t size, uintptr_t dev, bool *destroyed = nullptr)
      : buf(size), dev(dev), destroyed(destroyed) {}
  ~BufferRegion() override {
    if (destroyed)
      *destroyed = true;
  }
  void *getPtr() const override { return const_cast<uint8_t *>(buf.data()); }
  void *getDevicePtr() const override { return reinterpret_cast<void *>(dev); }
  std::size_t getSize() const override { return buf.size(); }
  std::vector<uint8_t> buf;
  uintptr_t dev;
  bool *destroyed;
};

static const void *at(uintptr_t addr) {
  return reinterpret_cast<const void *>(addr);
}

TEST(HostMemRegionTest, GetDeviceAddress) {
  FakeRegion r(0x1000, 0xA000'0000, 0x100);
  EXPECT_EQ(r.getDeviceAddress(at(0x1000), 0x100), 0xA000'0000u);
  EXPECT_EQ(r.getDeviceAddress(at(0x10FF), 1), 0xA000'00FFu);
  // Empty, starting before, ending past, starting past, and wrapping ranges.
  EXPECT_FALSE(r.getDeviceAddress(at(0x1000), 0));
  EXPECT_FALSE(r.getDeviceAddress(at(0x0FFF), 2));
  EXPECT_FALSE(r.getDeviceAddress(at(0x10FF), 2));
  EXPECT_FALSE(r.getDeviceAddress(at(0x1100), 1));
  EXPECT_FALSE(r.getDeviceAddress(at(0x1080), SIZE_MAX));

  constexpr uintptr_t maxPtr = std::numeric_limits<uintptr_t>::max();
  // Host range at the top of the address space.
  FakeRegion top(maxPtr - 0xF, 0x2000, 0x10);
  EXPECT_EQ(top.getDeviceAddress(at(maxPtr - 0x7), 8), 0x2008u);
  EXPECT_FALSE(top.getDeviceAddress(at(maxPtr - 0x7), 9));

  // Device range at the top of the address space.
  if constexpr (sizeof(uintptr_t) == sizeof(uint64_t)) {
    FakeRegion devTop(0x1000, maxPtr - 0x3, 0x10);
    EXPECT_EQ(devTop.getDeviceAddress(at(0x1000), 4), maxPtr - 0x3);
    EXPECT_FALSE(devTop.getDeviceAddress(at(0x1000), 5));
    EXPECT_FALSE(devTop.getDeviceAddress(at(0x1004), 1));
  }
}

TEST(SegmentTest, DeviceAddress) {
  uint8_t bytes[4] = {};
  EXPECT_FALSE((Segment{bytes, sizeof(bytes)}.getDeviceAddress()));

  BufferRegion region(16, 0x8000'0000);
  const uint8_t *data = region.buf.data();
  EXPECT_EQ((Segment{data + 8, 4, &region}.getDeviceAddress()), 0x8000'0008u);
  EXPECT_FALSE((Segment{data + 14, 4, &region}.getDeviceAddress()));
}

/// A header with no region, then two 4-byte segments sharing one region which
/// the message owns and hands over via take() after both have been taken.
struct TestMessage : public SegmentedMessageData {
  TestMessage(bool *destroyed = nullptr)
      : header{0xA0, 0xA1},
        region(std::make_unique<BufferRegion>(8, 0xD000'0000, destroyed)) {
    base = static_cast<const uint8_t *>(region->getPtr());
  }

  size_t numSegments() const override { return 3; }
  Segment segment(size_t idx) const override {
    if (taken.at(idx))
      throw std::runtime_error("segment has been taken");
    if (idx == 0)
      return {header.data(), header.size()};
    return {base + (idx - 1) * 4, 4, region.get()};
  }
  std::unique_ptr<HostMemRegion> take(size_t idx) override {
    if (idx == 0)
      return nullptr;
    if (taken.at(idx))
      throw std::runtime_error("segment has been taken");
    taken[idx] = true;
    return taken[1] && taken[2] ? std::move(region) : nullptr;
  }

  std::vector<uint8_t> header;
  std::unique_ptr<HostMemRegion> region;
  const uint8_t *base;
  std::vector<bool> taken = {false, false, false};
};

TEST(SegmentTest, CursorRemainingSegmentKeepsRegion) {
  TestMessage msg;
  SegmentedMessageDataCursor cursor(msg);
  EXPECT_EQ(cursor.remainingSegment().region, nullptr);

  // A partial advance keeps the region; the device address follows the data.
  cursor.advance(3);
  Segment s = cursor.remainingSegment();
  EXPECT_EQ(s.data, msg.base + 1);
  EXPECT_EQ(s.size, 3u);
  EXPECT_EQ(s.region, msg.region.get());
  EXPECT_EQ(s.getDeviceAddress(), 0xD000'0001u);
  EXPECT_EQ(cursor.remaining().data(), s.data);

  cursor.advance(3);
  EXPECT_EQ(cursor.remainingSegment().getDeviceAddress(), 0xD000'0004u);
  cursor.advance(4);
  EXPECT_TRUE(cursor.done());
  EXPECT_EQ(cursor.remainingSegment().data, nullptr);
}

TEST(SegmentTest, TakeHandsOverSharedRegionAfterLastUser) {
  bool destroyed = false;
  auto msg = std::make_unique<TestMessage>(&destroyed);
  HostMemRegion *shared = msg->region.get();

  // Segments without a region are unaffected by take().
  EXPECT_EQ(msg->take(0), nullptr);
  EXPECT_EQ(msg->segment(0).size, 2u);

  // Taking the first user marks it taken but keeps the shared region.
  EXPECT_EQ(msg->take(1), nullptr);
  EXPECT_THROW(msg->segment(1), std::runtime_error);
  EXPECT_THROW(msg->take(1), std::runtime_error);
  EXPECT_THROW(msg->toMessageData(), std::runtime_error);
  EXPECT_EQ(msg->segment(2).region, shared);

  // A cursor created after a take() throws upon reaching the taken segment.
  SegmentedMessageDataCursor cursor(*msg);
  EXPECT_THROW(cursor.advance(3), std::runtime_error);

  // Taking the last user hands over the region, which outlives the message.
  std::unique_ptr<HostMemRegion> region = msg->take(2);
  EXPECT_EQ(region.get(), shared);
  msg.reset();
  EXPECT_FALSE(destroyed);
  region.reset();
  EXPECT_TRUE(destroyed);

  // An untaken region is freed with its message.
  destroyed = false;
  { TestMessage untaken(&destroyed); }
  EXPECT_TRUE(destroyed);
}

TEST(SegmentTest, MessageDataHasNoRegions) {
  MessageData md(std::vector<uint8_t>{1, 2, 3});
  EXPECT_EQ(md.segment(0).region, nullptr);
  EXPECT_EQ(md.take(0), nullptr);
  EXPECT_EQ(md.segment(0).data, md.getBytes());
}

} // namespace
