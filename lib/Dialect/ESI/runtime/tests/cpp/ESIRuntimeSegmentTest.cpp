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
static_assert(
    std::is_same_v<HostMem::Options, services::HostMemAllocator::Options>);
static_assert(std::is_aggregate_v<Segment>);
static_assert(std::is_trivially_copyable_v<Segment>);

/// A HostMemRegion whose host and device addresses are arbitrary integers. The
/// memory is never dereferenced through these pointers.
struct FakeRegion : public HostMem::HostMemRegion {
  FakeRegion(uintptr_t hostBase, uintptr_t devBase, std::size_t size)
      : hostBase(hostBase), devBase(devBase), size(size) {}
  void *getPtr() const override { return reinterpret_cast<void *>(hostBase); }
  void *getDevicePtr() const override {
    return reinterpret_cast<void *>(devBase);
  }
  std::size_t getSize() const override { return size; }

  uintptr_t hostBase;
  uintptr_t devBase;
  std::size_t size;
};

/// A HostMemRegion backed by a real host buffer, with a distinct device base.
/// Sets `*destroyed` when deconstructed.
struct BufferRegion : public HostMem::HostMemRegion {
  BufferRegion(std::size_t size, uintptr_t devBase, bool *destroyed = nullptr)
      : buf(size), devBase(devBase), destroyed(destroyed) {}
  ~BufferRegion() override {
    if (destroyed)
      *destroyed = true;
  }
  void *getPtr() const override { return const_cast<uint8_t *>(buf.data()); }
  void *getDevicePtr() const override {
    return reinterpret_cast<void *>(devBase);
  }
  std::size_t getSize() const override { return buf.size(); }

  std::vector<uint8_t> buf;
  uintptr_t devBase;
  bool *destroyed;
};

static const void *at(uintptr_t addr) {
  return reinterpret_cast<const void *>(addr);
}

static Segment makeSegment(const uint8_t *data, size_t size) {
  return {data, size};
}

TEST(SegmentTest, NoRegionByDefault) {
  uint8_t bytes[4] = {1, 2, 3, 4};
  Segment a{bytes, sizeof(bytes)};
  Segment b = makeSegment(bytes, 2);
  EXPECT_EQ(a.region, nullptr);
  EXPECT_FALSE(a.getDeviceAddress().has_value());
  EXPECT_EQ(b.region, nullptr);
  EXPECT_EQ(b.span().size(), 2u);
}

TEST(SegmentTest, RegionDeviceAddress) {
  BufferRegion region(64, 0x8000'0000);
  ASSERT_NE(region.getPtr(), region.getDevicePtr());
  const uint8_t *data = region.buf.data();

  Segment seg{data + 8, 16, &region};
  EXPECT_EQ(seg.region, &region);
  EXPECT_EQ(seg.getDeviceAddress(), 0x8000'0000u + 8);

  // Copies are views of the same region.
  Segment copy = seg;
  EXPECT_EQ(copy.region, &region);

  // A segment whose bytes are not inside its region has no device address.
  Segment oob{data + 60, 8, &region};
  EXPECT_FALSE(oob.getDeviceAddress().has_value());
}

TEST(HostMemRegionTest, GetDeviceAddressInBounds) {
  FakeRegion region(0x1000, 0xA000'0000, 0x100);
  // Start, middle, and the last byte.
  EXPECT_EQ(region.getDeviceAddress(at(0x1000), 0x100), 0xA000'0000u);
  EXPECT_EQ(region.getDeviceAddress(at(0x1000), 1), 0xA000'0000u);
  EXPECT_EQ(region.getDeviceAddress(at(0x1080), 0x10), 0xA000'0080u);
  EXPECT_EQ(region.getDeviceAddress(at(0x10FF), 1), 0xA000'00FFu);
}

TEST(HostMemRegionTest, GetDeviceAddressOutOfBounds) {
  FakeRegion region(0x1000, 0xA000'0000, 0x100);
  // Zero size.
  EXPECT_FALSE(region.getDeviceAddress(at(0x1000), 0).has_value());
  // Starts before the region.
  EXPECT_FALSE(region.getDeviceAddress(at(0x0FFF), 2).has_value());
  EXPECT_FALSE(region.getDeviceAddress(at(0x0F00), 0x10).has_value());
  // Ends one byte past the region.
  EXPECT_FALSE(region.getDeviceAddress(at(0x1000), 0x101).has_value());
  EXPECT_FALSE(region.getDeviceAddress(at(0x10FF), 2).has_value());
  // Starts at or after the end.
  EXPECT_FALSE(region.getDeviceAddress(at(0x1100), 1).has_value());
  // Size which would wrap the address space.
  EXPECT_FALSE(
      region
          .getDeviceAddress(at(0x1080), std::numeric_limits<std::size_t>::max())
          .has_value());
}

TEST(HostMemRegionTest, GetDeviceAddressOverflow) {
  constexpr uintptr_t maxPtr = std::numeric_limits<uintptr_t>::max();
  // A region at the very top of the address space: ptr + size would overflow.
  FakeRegion top(maxPtr - 0xF, 0x2000, 0x10);
  EXPECT_EQ(top.getDeviceAddress(at(maxPtr - 0x7), 8), 0x2008u);
  EXPECT_FALSE(top.getDeviceAddress(at(maxPtr - 0x7), 9).has_value());
  EXPECT_FALSE(top.getDeviceAddress(at(maxPtr), 2).has_value());
  EXPECT_FALSE(top.getDeviceAddress(at(maxPtr - 0x7),
                                    std::numeric_limits<std::size_t>::max() - 4)
                   .has_value());

  // A device base where devBase + offset would overflow.
  if constexpr (sizeof(uintptr_t) == sizeof(uint64_t)) {
    FakeRegion devTop(0x1000, maxPtr - 0x3, 0x10);
    EXPECT_EQ(devTop.getDeviceAddress(at(0x1000), 1), maxPtr - 0x3);
    EXPECT_EQ(devTop.getDeviceAddress(at(0x1003), 1), maxPtr);
    EXPECT_FALSE(devTop.getDeviceAddress(at(0x1004), 1).has_value());
  }
}

/// Header and footer owned directly; payload in a HostMem region owned by the
/// message and handed over by take().
struct ThreeSegmentMessage : public SegmentedMessageData {
  ThreeSegmentMessage(bool *regionDestroyed = nullptr)
      : header{0xA0, 0xA1}, footer{0xF0, 0xF1, 0xF2} {
    auto r = std::make_unique<BufferRegion>(8, 0xC000'0000, regionDestroyed);
    for (size_t i = 0; i < r->buf.size(); ++i)
      r->buf[i] = static_cast<uint8_t>(0x10 + i);
    payloadData = r->buf.data();
    payloadSize = r->buf.size();
    payload = std::move(r);
  }

  size_t numSegments() const override { return 3; }
  Segment segment(size_t idx) const override {
    switch (idx) {
    case 0:
      return {header.data(), header.size()};
    case 1:
      if (!payload)
        throw std::runtime_error("segment 1 has been taken");
      return {payloadData, payloadSize, payload.get()};
    case 2:
      return {footer.data(), footer.size()};
    default:
      throw std::out_of_range("ThreeSegmentMessage has 3 segments");
    }
  }
  std::unique_ptr<HostMemRegion> take(size_t segIdx) override {
    if (segIdx != 1)
      return nullptr;
    if (!payload)
      throw std::runtime_error("segment 1 has been taken");
    return std::move(payload);
  }

  std::vector<uint8_t> header;
  std::vector<uint8_t> footer;
  std::unique_ptr<HostMemRegion> payload;
  const uint8_t *payloadData;
  size_t payloadSize;
};

TEST(SegmentTest, MessageSegmentsReferenceRegion) {
  ThreeSegmentMessage msg;
  EXPECT_EQ(msg.segment(0).region, nullptr);
  EXPECT_EQ(msg.segment(1).region, msg.payload.get());
  EXPECT_EQ(msg.segment(1).getDeviceAddress(), 0xC000'0000u);
  EXPECT_EQ(msg.segment(2).region, nullptr);

  MessageData flat = msg.toMessageData();
  std::vector<uint8_t> expected = {0xA0, 0xA1, 0x10, 0x11, 0x12, 0x13, 0x14,
                                   0x15, 0x16, 0x17, 0xF0, 0xF1, 0xF2};
  EXPECT_EQ(flat.getData(), expected);

  // The cursor walks region-backed segments like any other.
  SegmentedMessageDataCursor cursor(msg);
  cursor.advance(3);
  EXPECT_EQ(cursor.remaining().data(), msg.payloadData + 1);
}

TEST(SegmentTest, EngineTakesRegionsAndReturnsToPool) {
  bool destroyed = false;
  auto msg = std::make_unique<ThreeSegmentMessage>(&destroyed);
  HostMemRegion *payload = msg->payload.get();
  std::vector<std::unique_ptr<HostMemRegion>> pool;
  std::vector<uint8_t> bounced;
  std::vector<uint64_t> dmaAddrs;

  // Mimic a scatter-gather engine's write path.
  for (size_t i = 0; i < msg->numSegments(); ++i) {
    Segment s = msg->segment(i);
    if (std::optional<uint64_t> dev = s.getDeviceAddress()) {
      dmaAddrs.push_back(*dev);
      // Once transmitted, take the region and return it to the pool.
      pool.push_back(msg->take(i));
    } else {
      bounced.insert(bounced.end(), s.data, s.data + s.size);
    }
  }
  EXPECT_EQ(dmaAddrs, std::vector<uint64_t>{0xC000'0000u});
  EXPECT_EQ(bounced, (std::vector<uint8_t>{0xA0, 0xA1, 0xF0, 0xF1, 0xF2}));
  ASSERT_EQ(pool.size(), 1u);
  EXPECT_EQ(pool[0].get(), payload);

  // The taken segment is no longer accessible; the others still are.
  EXPECT_THROW(msg->segment(1), std::runtime_error);
  EXPECT_THROW(msg->take(1), std::runtime_error);
  EXPECT_THROW(msg->toMessageData(), std::runtime_error);
  EXPECT_EQ(msg->segment(0).size, 2u);
  EXPECT_EQ(msg->segment(2).size, 3u);

  // Destroying the message doesn't free the region, which the pool now owns.
  msg.reset();
  EXPECT_FALSE(destroyed);
  EXPECT_EQ(static_cast<BufferRegion *>(pool[0].get())->buf[0], 0x10);
  pool.clear();
  EXPECT_TRUE(destroyed);
}

TEST(SegmentTest, TakingUnbackedSegmentLeavesItAccessible) {
  bool destroyed = false;
  {
    ThreeSegmentMessage msg(&destroyed);
    EXPECT_EQ(msg.take(0), nullptr);
    EXPECT_EQ(msg.segment(0).size, 2u);
  }
  // The untaken region is freed with the message.
  EXPECT_TRUE(destroyed);
}

TEST(SegmentTest, MessageDataHasNoRegions) {
  MessageData md(std::vector<uint8_t>{1, 2, 3});
  EXPECT_EQ(md.segment(0).region, nullptr);
  EXPECT_EQ(md.take(0), nullptr);
  EXPECT_EQ(md.segment(0).data, md.getBytes());
}

} // namespace
