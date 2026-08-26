//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AXI4/AXI4Attributes.h"
#include "circt/Dialect/AXI4/AXI4Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace axi4;

namespace {

class WindowSetTest : public testing::Test {
protected:
  void SetUp() override { context.loadDialect<AXI4Dialect>(); }

  BurstSetAttr burstSet(ArrayRef<std::pair<BurstKind, uint32_t>> specs) {
    SmallVector<BurstSpecAttr> attrs;
    for (auto [kind, len] : specs)
      attrs.push_back(BurstSpecAttr::get(&context, kind, len));
    return BurstSetAttr::get(&context, attrs);
  }

  WindowAttr window(uint64_t base, uint64_t last,
                    ArrayRef<std::pair<BurstKind, uint32_t>> specs) {
    return WindowAttr::get(&context, base, last, burstSet(specs));
  }

  MLIRContext context;
};

// Ensure that window_sets with the same windows given in a different order are
// equivalent
TEST_F(WindowSetTest, ShuffledInputUniquesToSortedInput) {
  auto low = window(0, 0xFF, {{BurstKind::Fixed, 4}});
  auto high = window(0x1000, 0x10FF, {{BurstKind::Incr, 8}});

  WindowAttr sorted[] = {low, high};
  WindowAttr shuffled[] = {high, low};

  auto a = WindowSetAttr::get(&context, sorted);
  auto b = WindowSetAttr::get(&context, shuffled);

  EXPECT_EQ(a, b);
  EXPECT_EQ(a.getAsOpaquePointer(), b.getAsOpaquePointer());
  EXPECT_EQ(a.getWindows().size(), 2u);
}

// Ensure that different spellings of the same capability map are equivalent
TEST_F(WindowSetTest, OverlappingAndDisjointSpellingsAreEquivalent) {
  WindowAttr overlapping[] = {window(0, 0xFFF, {{BurstKind::Fixed, 4}}),
                              window(0, 0xFF, {{BurstKind::Incr, 8}})};
  WindowAttr disjoint[] = {
      window(0, 0xFF, {{BurstKind::Fixed, 4}, {BurstKind::Incr, 8}}),
      window(0x100, 0xFFF, {{BurstKind::Fixed, 4}})};

  auto a = WindowSetAttr::get(&context, overlapping);
  auto b = WindowSetAttr::get(&context, disjoint);

  EXPECT_EQ(a, b);
  ASSERT_EQ(a.getWindows().size(), 2u);
  EXPECT_EQ(a.getWindows()[0], disjoint[0]);
  EXPECT_EQ(a.getWindows()[1], disjoint[1]);
}

// Check a window subsumed by a window with the same burst kind is absorbed by
// it, leaving one window rather than two
TEST_F(WindowSetTest, OverlapAbsorbedByLongerBurst) {
  WindowAttr overlapping[] = {window(0, 0xFF, {{BurstKind::Incr, 4}}),
                              window(0, 0x1FF, {{BurstKind::Incr, 16}})};

  auto set = WindowSetAttr::get(&context, overlapping);

  ASSERT_EQ(set.getWindows().size(), 1u);
  EXPECT_EQ(set.getWindows()[0], window(0, 0x1FF, {{BurstKind::Incr, 16}}));
}

// Ensure that window_sets covering the whole address space are equivalent
// wherever they happen to be split
TEST_F(WindowSetTest, WholeAddressSpaceSplitsAreEquivalent) {
  WindowAttr splitHigh[] = {
      window(0, UINT64_MAX - 1, {{BurstKind::Fixed, 4}}),
      window(UINT64_MAX, UINT64_MAX, {{BurstKind::Fixed, 4}})};
  WindowAttr splitMiddle[] = {
      window(0, 0x7FFFFFFFFFFFFFFF, {{BurstKind::Fixed, 4}}),
      window(0x8000000000000000, UINT64_MAX, {{BurstKind::Fixed, 4}})};

  auto a = WindowSetAttr::get(&context, splitHigh);
  auto b = WindowSetAttr::get(&context, splitMiddle);

  EXPECT_EQ(a, b);
  ASSERT_EQ(a.getWindows().size(), 1u);
  EXPECT_EQ(a.getWindows()[0], window(0, UINT64_MAX, {{BurstKind::Fixed, 4}}));
}

} // namespace
