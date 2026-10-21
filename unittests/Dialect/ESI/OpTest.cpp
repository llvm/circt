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
  auto typeAliasRef =
      SymbolRefAttr::get(b.getStringAttr("types"),
                         {FlatSymbolRefAttr::get(b.getStringAttr("foo"))});
  auto typeAliasAny = b.getType<hw::TypeAliasType>(
      typeAliasRef, b.getType<AnyType>(), b.getType<AnyType>());
  EXPECT_SUCCESS(checkInnerTypeMatch(typeAliasAny, i1Type));
}

TEST(ESIOpTest, ReplaceBundleSubElements) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect, ESIDialect>();
  Builder builder(&context);

  auto i8Type = builder.getI8Type();
  auto name = builder.getStringAttr("payload");
  auto ref = SymbolRefAttr::get(
      builder.getStringAttr("types"),
      {FlatSymbolRefAttr::get(builder.getStringAttr("payloadAlias"))});
  auto alias = hw::TypeAliasType::get(ref, i8Type);
  auto channel = ChannelType::get(&context, alias, ChannelSignaling::FIFO, 2);
  auto expectedChannel =
      ChannelType::get(&context, i8Type, ChannelSignaling::FIFO, 2);
  auto bundle = ChannelBundleType::get(
      &context, {{name, ChannelDirection::to, channel}}, builder.getUnitAttr());

  AttrTypeReplacer replacer;
  replacer.addReplacement(
      [](hw::TypeAliasType type) { return type.getCanonicalType(); });
  replacer.addReplacement([&](StringAttr attr) -> std::optional<Attribute> {
    if (attr == name)
      return builder.getStringAttr("renamed");
    return std::nullopt;
  });
  auto renamedBundle = cast<ChannelBundleType>(replacer.replace(bundle));
  EXPECT_EQ(renamedBundle.getResettable(), bundle.getResettable());
  ASSERT_EQ(renamedBundle.getChannels().size(), 1u);
  EXPECT_EQ(renamedBundle.getChannels()[0].name,
            builder.getStringAttr("renamed"));
  EXPECT_EQ(renamedBundle.getChannels()[0].direction, ChannelDirection::to);
  EXPECT_EQ(renamedBundle.getChannels()[0].type, expectedChannel);
}

TEST(ESIOpTest, CanonicalizeNestedAliases) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect, ESIDialect>();
  Builder builder(&context);

  auto i8Type = builder.getI8Type();
  auto name = builder.getStringAttr("payload");
  auto ref = SymbolRefAttr::get(
      builder.getStringAttr("types"),
      {FlatSymbolRefAttr::get(builder.getStringAttr("payloadAlias"))});
  auto alias = hw::TypeAliasType::get(ref, i8Type);
  auto channel = ChannelType::get(&context, alias, ChannelSignaling::FIFO, 2);
  auto expectedChannel =
      ChannelType::get(&context, i8Type, ChannelSignaling::FIFO, 2);
  EXPECT_EQ(hw::getCanonicalType(channel), expectedChannel);
  EXPECT_EQ(hw::getCanonicalType(ListType::get(&context, alias)),
            ListType::get(&context, i8Type));

  auto bundle = ChannelBundleType::get(
      &context, {{name, ChannelDirection::to, channel}}, builder.getUnitAttr());
  auto canonicalBundle = cast<ChannelBundleType>(hw::getCanonicalType(bundle));
  EXPECT_EQ(canonicalBundle.getResettable(), bundle.getResettable());
  ASSERT_EQ(canonicalBundle.getChannels().size(), 1u);
  EXPECT_EQ(canonicalBundle.getChannels()[0].name, name);
  EXPECT_EQ(canonicalBundle.getChannels()[0].direction, ChannelDirection::to);
  EXPECT_EQ(canonicalBundle.getChannels()[0].type, expectedChannel);
  auto bundleAlias = hw::TypeAliasType::get(
      SymbolRefAttr::get(
          builder.getStringAttr("types"),
          {FlatSymbolRefAttr::get(builder.getStringAttr("bundleAlias"))}),
      bundle);
  EXPECT_EQ(bundleAlias.getCanonicalType(), canonicalBundle);

  auto field = hw::StructType::get(&context, {{name, alias}});
  auto frame = WindowFrameType::get(
      &context, name, {WindowFieldType::get(&context, name, 0, 0)});
  auto window = WindowType::get(&context, builder.getStringAttr("window"),
                                field, {frame});
  auto canonicalWindow = cast<WindowType>(hw::getCanonicalType(window));
  EXPECT_EQ(canonicalWindow.getName(), window.getName());
  EXPECT_EQ(canonicalWindow.getFrames(), window.getFrames());
  EXPECT_EQ(canonicalWindow.getInto(),
            hw::StructType::get(&context, {{name, i8Type}}));
}
} // namespace
