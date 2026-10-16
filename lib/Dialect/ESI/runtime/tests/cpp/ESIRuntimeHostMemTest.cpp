//===- ESIRuntimeHostMemTest.cpp - Host memory allocation tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "esi/Services.h"
#include "esi/Utils.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>

using namespace esi;

namespace {

TEST(HostMemAllocationTest, DefaultOptions) {
  services::HostMem::Options opts;
  EXPECT_EQ(opts.alignment, 0u);
  EXPECT_FALSE(opts.writeable);
  EXPECT_FALSE(opts.useLargePages);
}

class HostMemAlignedAllocationTest : public testing::TestWithParam<size_t> {};

TEST_P(HostMemAlignedAllocationTest, AllocateAndFree) {
  size_t alignment = GetParam();
  for (size_t size : {1, 7, 33, 4097}) {
    SCOPED_TRACE(size);
    auto deleter = [alignment](void *ptr) {
      utils::alignedFree(ptr, alignment);
    };
    std::unique_ptr<void, decltype(deleter)> buffer(
        utils::alignedAlloc(size, alignment), deleter);
    ASSERT_NE(buffer, nullptr);
    if (alignment)
      EXPECT_EQ(reinterpret_cast<uintptr_t>(buffer.get()) % alignment, 0u);
    else
      EXPECT_EQ(reinterpret_cast<uintptr_t>(buffer.get()) %
                    alignof(std::max_align_t),
                0u);
    std::memset(buffer.get(), 0xA5, size);
    for (size_t i = 0; i < size; ++i)
      EXPECT_EQ(static_cast<uint8_t *>(buffer.get())[i], 0xA5);
  }
  EXPECT_EQ(utils::alignedAlloc(0, alignment), nullptr);
  utils::alignedFree(nullptr, alignment);
}

INSTANTIATE_TEST_SUITE_P(Alignments, HostMemAlignedAllocationTest,
                         testing::Values(0, 1, 2, 8, 16, 32, 64, 4096, 8192));

TEST(HostMemAllocationTest, InvalidAlignment) {
  for (size_t alignment : {3, 6, 24, 4095})
    EXPECT_EQ(utils::alignedAlloc(17, alignment), nullptr);
}

TEST(HostMemAllocationTest, RoundingOverflow) {
  constexpr size_t max = std::numeric_limits<size_t>::max();
  // Even small requested alignments are promoted to a platform-safe alignment.
  for (size_t alignment : {1, 2, 8, 16, 32, 64, 4096, 8192})
    EXPECT_EQ(utils::alignedAlloc(max, alignment), nullptr);
  EXPECT_EQ(utils::alignedAlloc(max - 15, 32), nullptr);
  EXPECT_EQ(utils::alignedAlloc(max - 4095, 8192), nullptr);
}

} // namespace
