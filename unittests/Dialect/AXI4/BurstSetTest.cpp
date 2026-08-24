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

class BurstSetTest : public testing::Test {
protected:
  void SetUp() override { context.loadDialect<AXI4Dialect>(); }

  BurstSpecAttr spec(BurstKind kind, uint32_t len) {
    return BurstSpecAttr::get(&context, kind, len);
  }

  MLIRContext context;
};

// Ensure that burst_sets with the same specs given in a different order are
// equivalent
TEST_F(BurstSetTest, ShuffledInputUniquesToSortedInput) {
  auto fixed4 = spec(BurstKind::Fixed, 4);
  auto incr8 = spec(BurstKind::Incr, 8);
  auto wrap2 = spec(BurstKind::Wrap, 2);

  BurstSpecAttr sorted[] = {fixed4, incr8, wrap2};
  BurstSpecAttr shuffled[] = {wrap2, fixed4, incr8};

  auto a = BurstSetAttr::get(&context, sorted);
  auto b = BurstSetAttr::get(&context, shuffled);

  EXPECT_EQ(a, b);
  EXPECT_EQ(a.getAsOpaquePointer(), b.getAsOpaquePointer());
  EXPECT_EQ(a.getBurstSpecs().size(), 3u);
}

// Ensure burst_specs are accessed in the canonical order
TEST_F(BurstSetTest, StoredOrderIsCanonical) {
  BurstSpecAttr shuffled[] = {
      spec(BurstKind::Wrap, 2), spec(BurstKind::Incr, 256),
      spec(BurstKind::Incr, 1), spec(BurstKind::Fixed, 16)};
  auto set = BurstSetAttr::get(&context, shuffled);

  ASSERT_EQ(set.getBurstSpecs().size(), 4u);
  EXPECT_EQ(set.getBurstSpecs()[0], spec(BurstKind::Fixed, 16));
  EXPECT_EQ(set.getBurstSpecs()[1], spec(BurstKind::Incr, 1));
  EXPECT_EQ(set.getBurstSpecs()[2], spec(BurstKind::Incr, 256));
  EXPECT_EQ(set.getBurstSpecs()[3], spec(BurstKind::Wrap, 2));
}

// Check duplicate specs are collapsed together
TEST_F(BurstSetTest, DuplicatesCollapse) {
  auto incr8 = spec(BurstKind::Incr, 8);
  BurstSpecAttr withDuplicates[] = {incr8, incr8, incr8};

  auto set = BurstSetAttr::get(&context, withDuplicates);

  EXPECT_EQ(set.getBurstSpecs().size(), 1u);
  EXPECT_EQ(set, BurstSetAttr::get(&context, {incr8}));
}

} // namespace
