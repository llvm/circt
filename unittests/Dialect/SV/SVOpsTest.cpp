//===- SVOpsTest.cpp - SV op unit tests -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace sv;

namespace {

TEST(PackageOpTest, ResolveTypeAliases) {
  MLIRContext context;
  context.loadDialect<SVDialect, HWDialect>();
  auto loc = UnknownLoc::get(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module->getBody());

  auto package = PackageOp::create(builder, "types", StringAttr());
  auto scope = TypeScopeOp::create(builder, "legacy", StringAttr());
  auto interface = InterfaceOp::create(builder, "not_a_type_scope");
  auto i8 = builder.getIntegerType(8);
  auto i16 = builder.getIntegerType(16);

  builder.createBlock(&package.getBody());
  auto packageDecl =
      TypedeclOp::create(builder, builder.getStringAttr("word"), StringAttr(),
                         TypeAttr::get(i8), builder.getStringAttr("word_t"));
  builder.createBlock(&scope.getBody());
  auto scopeDecl =
      TypedeclOp::create(builder, builder.getStringAttr("word"), StringAttr(),
                         TypeAttr::get(i16), StringAttr());

  HWSymbolCache cache;
  cache.addDefinition(package.getSymNameAttr(), package);
  cache.addDefinition(scope.getSymNameAttr(), scope);
  cache.addDefinition(interface.getSymNameAttr(), interface);
  cache.freeze();

  auto alias = [&](StringRef root, StringRef leaf, Type inner) {
    return TypeAliasType::get(
        SymbolRefAttr::get(builder.getStringAttr(root),
                           {FlatSymbolRefAttr::get(&context, leaf)}),
        inner);
  };
  auto packageAlias = alias("types", "word", i8);
  auto scopeAlias = alias("legacy", "word", i16);
  EXPECT_EQ(packageDecl.getAliasType(), packageAlias);
  EXPECT_EQ(scopeDecl.getAliasType(), scopeAlias);
  EXPECT_EQ(packageAlias.getTypeDecl(cache), packageDecl);
  EXPECT_EQ(scopeAlias.getTypeDecl(cache), scopeDecl);

  EXPECT_FALSE(alias("missing", "word", i8).getTypeDecl(cache));
  EXPECT_FALSE(alias("types", "missing", i8).getTypeDecl(cache));
  EXPECT_FALSE(alias("types", "word_t", i8).getTypeDecl(cache));
  EXPECT_FALSE(alias("not_a_type_scope", "word", i8).getTypeDecl(cache));
  EXPECT_FALSE(TypeAliasType::get(FlatSymbolRefAttr::get(&context, "types"), i8)
                   .getTypeDecl(cache));
  EXPECT_FALSE(
      TypeAliasType::get(
          SymbolRefAttr::get(package.getSymNameAttr(),
                             {FlatSymbolRefAttr::get(&context, "nested"),
                              FlatSymbolRefAttr::get(&context, "word")}),
          i8)
          .getTypeDecl(cache));
}

TEST(SVVerbatimModuleOpTest, GetPortListArgNumIsDirectionRelative) {
  MLIRContext context;
  context.loadDialect<SVDialect, HWDialect>();
  LocationAttr loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto i1 = builder.getIntegerType(1);
  SmallVector<PortInfo> ports;
  ports.push_back(
      {{builder.getStringAttr("in0"), i1, ModulePort::Direction::Input}});
  ports.push_back(
      {{builder.getStringAttr("out0"), i1, ModulePort::Direction::Output}});
  ports.push_back(
      {{builder.getStringAttr("in1"), i1, ModulePort::Direction::Input}});
  ports.push_back(
      {{builder.getStringAttr("out1"), i1, ModulePort::Direction::Output}});
  ports.push_back(
      {{builder.getStringAttr("out2"), i1, ModulePort::Direction::Output}});

  auto verbatimModule = SVVerbatimModuleOp::create(
      builder, builder.getStringAttr("Top"), ports,
      FlatSymbolRefAttr::get(builder.getStringAttr("source")));

  auto portList = verbatimModule.getPortList();
  ASSERT_EQ(portList.size(), 5u);

  EXPECT_EQ(portList[0].name, builder.getStringAttr("in0"));
  EXPECT_EQ(portList[0].argNum, 0u);

  EXPECT_EQ(portList[1].name, builder.getStringAttr("out0"));
  EXPECT_EQ(portList[1].argNum, 0u);

  EXPECT_EQ(portList[2].name, builder.getStringAttr("in1"));
  EXPECT_EQ(portList[2].argNum, 1u);

  EXPECT_EQ(portList[3].name, builder.getStringAttr("out1"));
  EXPECT_EQ(portList[3].argNum, 1u);

  EXPECT_EQ(portList[4].name, builder.getStringAttr("out2"));
  EXPECT_EQ(portList[4].argNum, 2u);
}

} // namespace
