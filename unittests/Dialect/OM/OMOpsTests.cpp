//===- OMOpsTests.cpp - OM operation tests --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt::om;

namespace {

class OMOpsTest : public ::testing::Test {
protected:
  OMOpsTest() : builder(&context) { context.getOrLoadDialect<om::OMDialect>(); }

  OwningOpRef<ModuleOp> createModule() {
    auto module = ModuleOp::create(UnknownLoc::get(&context));
    builder.setInsertionPointToStart(module.getBody());
    om::ClassOp::buildSimpleClassOp(
        builder, UnknownLoc::get(&context), "Test", {}, {"first", "second"},
        {builder.getI32Type(), builder.getI32Type()});
    return OwningOpRef<ModuleOp>(module);
  }

  MLIRContext context;
  OpBuilder builder;
};

TEST_F(OMOpsTest, UpdateFieldsUsesExplicitLocationsWithoutFusedMetadata) {
  auto module = createModule();
  auto classOp = cast<om::ClassOp>(module->getBody()->front());
  auto fieldsOp = classOp.getFieldsOp();

  Location firstLoc = NameLoc::get(builder.getStringAttr("first"));
  Location secondLoc = NameLoc::get(builder.getStringAttr("second"));
  Location newLoc = NameLoc::get(builder.getStringAttr("new"));
  SmallVector<Location> oldLocs{firstLoc, secondLoc};
  fieldsOp->setLoc(FusedLoc::get(&context, oldLocs));
  fieldsOp->setAttr(
      "field_locs",
      builder.getArrayAttr({LocationAttr(firstLoc), LocationAttr(secondLoc)}));

  SmallVector<Location> newLocs{newLoc};
  SmallVector<Value> newValues{fieldsOp.getFields().front()};
  SmallVector<Attribute> newNames{builder.getStringAttr("new")};
  classOp.updateFields(newLocs, newValues, newNames);

  auto fusedLoc = dyn_cast<FusedLoc>(classOp.getFieldsOp().getLoc());
  ASSERT_TRUE(fusedLoc);
  auto metadata = dyn_cast<ArrayAttr>(fusedLoc.getMetadata());
  ASSERT_TRUE(metadata);
  ASSERT_EQ(metadata.size(), 3u);
  EXPECT_EQ(cast<LocationAttr>(metadata[0]), LocationAttr(firstLoc));
  EXPECT_EQ(cast<LocationAttr>(metadata[1]), LocationAttr(secondLoc));
  EXPECT_EQ(cast<LocationAttr>(metadata[2]), LocationAttr(newLoc));
}

TEST_F(OMOpsTest, UpdateFieldsFallsBackToFieldsOpLocation) {
  auto module = createModule();
  auto classOp = cast<om::ClassOp>(module->getBody()->front());
  auto fieldsOp = classOp.getFieldsOp();

  SmallVector<Location> oldLocs{NameLoc::get(builder.getStringAttr("first")),
                                NameLoc::get(builder.getStringAttr("second"))};
  Location fieldsLoc = FusedLoc::get(&context, oldLocs);
  fieldsOp->setLoc(fieldsLoc);
  fieldsOp->removeAttr("field_locs");

  Location newLoc = NameLoc::get(builder.getStringAttr("new"));
  SmallVector<Location> newLocs{newLoc};
  SmallVector<Value> newValues{fieldsOp.getFields().front()};
  SmallVector<Attribute> newNames{builder.getStringAttr("new")};
  classOp.updateFields(newLocs, newValues, newNames);

  auto fusedLoc = dyn_cast<FusedLoc>(classOp.getFieldsOp().getLoc());
  ASSERT_TRUE(fusedLoc);
  auto metadata = dyn_cast<ArrayAttr>(fusedLoc.getMetadata());
  ASSERT_TRUE(metadata);
  ASSERT_EQ(metadata.size(), 3u);
  EXPECT_EQ(cast<LocationAttr>(metadata[0]), LocationAttr(fieldsLoc));
  EXPECT_EQ(cast<LocationAttr>(metadata[1]), LocationAttr(fieldsLoc));
  EXPECT_EQ(cast<LocationAttr>(metadata[2]), LocationAttr(newLoc));
}

} // namespace
