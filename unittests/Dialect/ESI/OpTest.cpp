//===- GraphFixutre.cpp - A fixture for instance graph unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace esi;

#define EXPECT_SUCCESS(expr) EXPECT_TRUE(succeeded(expr))
#define EXPECT_FAILURE(expr) EXPECT_FALSE(succeeded(expr))

namespace {
TEST(ESIOpTest, TypeMatching) {
  MLIRContext ctxt;
  ImplicitLocOpBuilder b(UnknownLoc::get(&ctxt), &ctxt);
  ctxt.loadDialect<hw::HWDialect>();
  ctxt.loadDialect<esi::ESIDialect>();
  auto aStr = b.getStringAttr("a");
  IntegerType i1Type = b.getI1Type();

  // Channel type tests
  EXPECT_SUCCESS(checkInnerTypeMatch(b.getType<ChannelType>(i1Type),
                                     b.getType<ChannelType>(i1Type)));
  EXPECT_FAILURE(checkInnerTypeMatch(b.getType<ChannelType>(i1Type), i1Type));

  // Any type tests
  EXPECT_SUCCESS(checkInnerTypeMatch(b.getType<AnyType>(), i1Type));

  // Struct type tests
  auto structAny = hw::StructType::get(&ctxt, {{aStr, b.getType<AnyType>()}});
  EXPECT_SUCCESS(checkInnerTypeMatch(
      structAny, hw::StructType::get(&ctxt, {{aStr, i1Type}})));
  EXPECT_FAILURE(checkInnerTypeMatch(structAny, i1Type));
  EXPECT_FAILURE(checkInnerTypeMatch(
      structAny, hw::StructType::get(
                     &ctxt, {{aStr, i1Type}, {b.getStringAttr("b"), i1Type}})));

  // Array type tests
  auto arrayAny = hw::ArrayType::get(b.getType<AnyType>(), 1);
  EXPECT_SUCCESS(checkInnerTypeMatch(arrayAny, hw::ArrayType::get(i1Type, 1)));
  EXPECT_FAILURE(checkInnerTypeMatch(arrayAny, i1Type));
  EXPECT_FAILURE(checkInnerTypeMatch(arrayAny, hw::ArrayType::get(i1Type, 2)));

  // Union type tests
  auto unionAny = hw::UnionType::get(&ctxt, {{aStr, b.getType<AnyType>(), 0}});
  EXPECT_SUCCESS(checkInnerTypeMatch(
      unionAny, hw::UnionType::get(&ctxt, {{aStr, i1Type, 0}})));
  EXPECT_FAILURE(checkInnerTypeMatch(unionAny, i1Type));
  EXPECT_FAILURE(checkInnerTypeMatch(
      unionAny, hw::UnionType::get(&ctxt, {{aStr, i1Type, 1}})));
  EXPECT_FAILURE(checkInnerTypeMatch(
      unionAny,
      hw::UnionType::get(
          &ctxt, {{aStr, i1Type, 0}, {b.getStringAttr("b"), i1Type, 1}})));

  // ESI list tests
  auto esiListAny = b.getType<ListType>(b.getType<AnyType>());
  EXPECT_FAILURE(checkInnerTypeMatch(esiListAny, i1Type));
  EXPECT_SUCCESS(checkInnerTypeMatch(esiListAny, b.getType<ListType>(i1Type)));

  // ESI window tests
  auto esiWindowAny = WindowType::get(
      &ctxt, b.getStringAttr("aWindow"), structAny,
      {WindowFrameType::get(&ctxt, aStr,
                            {WindowFieldType::get(&ctxt, aStr, 0, {})})});
  EXPECT_SUCCESS(checkInnerTypeMatch(
      esiWindowAny, hw::StructType::get(&ctxt, {{aStr, i1Type}})));
  EXPECT_SUCCESS(checkInnerTypeMatch(
      esiWindowAny,
      WindowType::get(&ctxt, b.getStringAttr("aWindow"),
                      hw::StructType::get(&ctxt, {{aStr, i1Type}}),
                      {WindowFrameType::get(&ctxt, aStr, {})})));
  EXPECT_FAILURE(checkInnerTypeMatch(esiWindowAny, i1Type));

  // Type alias type tests
  auto typeAliasAny =
      b.getType<hw::TypeAliasType>(b.getType<SymbolRefAttr>("foo"),
                                   b.getType<AnyType>(), b.getType<AnyType>());
  EXPECT_SUCCESS(checkInnerTypeMatch(typeAliasAny, i1Type));
}
} // namespace
