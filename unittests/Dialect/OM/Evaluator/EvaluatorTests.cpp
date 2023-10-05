//===- EvaluatorTest.cpp - Object Model evaluator tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/Evaluator/Evaluator.h"
#include "circt/Dialect/OM/OMAttributes.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMOps.h"
#include "circt/Dialect/OM/OMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"
#include <mlir/IR/BuiltinAttributes.h>

using namespace mlir;
using namespace circt::om;

namespace {

/// Failure scenarios.

TEST(EvaluatorTests, InstantiateInvalidClassName) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  Evaluator evaluator(mod);

  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_EQ(diag.str(), "unknown class name \"MyClass\"");
  });

  auto result = evaluator.instantiate(builder.getStringAttr("MyClass"), {});

  ASSERT_FALSE(succeeded(result));
}

TEST(EvaluatorTests, InstantiateInvalidParamSize) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  StringRef params[] = {"param"};
  auto cls = builder.create<ClassOp>("MyClass", params);
  cls.getBody().emplaceBlock().addArgument(
      circt::om::OMIntegerType::get(&context), cls.getLoc());

  Evaluator evaluator(mod);

  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_EQ(
        diag.str(),
        "actual parameter list length (0) does not match formal parameter "
        "list length (1)");
  });

  auto result = evaluator.instantiate(builder.getStringAttr("MyClass"), {});

  ASSERT_FALSE(succeeded(result));
}

TEST(EvaluatorTests, InstantiateNullParam) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  StringRef params[] = {"param"};
  auto cls = builder.create<ClassOp>("MyClass", params);
  cls.getBody().emplaceBlock().addArgument(
      circt::om::OMIntegerType::get(&context), cls.getLoc());

  Evaluator evaluator(mod);

  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_EQ(diag.str(), "actual parameter for \"param\" is null");
  });

  auto result =
      evaluator.instantiate(builder.getStringAttr("MyClass"), {nullptr});

  ASSERT_FALSE(succeeded(result));
}

TEST(EvaluatorTests, InstantiateInvalidParamType) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  StringRef params[] = {"param"};
  auto cls = builder.create<ClassOp>("MyClass", params);
  cls.getBody().emplaceBlock().addArgument(
      circt::om::OMIntegerType::get(&context), cls.getLoc());

  Evaluator evaluator(mod);

  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_EQ(diag.str(), "actual parameter for \"param\" has invalid type");
  });

  auto result =
      evaluator.instantiate(builder.getStringAttr("MyClass"),
                            getEvaluatorValuesFromAttributes(
                                &context, {builder.getF32FloatAttr(42)}));

  ASSERT_FALSE(succeeded(result));
}

TEST(EvaluatorTests, GetFieldInvalidName) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  auto cls = builder.create<ClassOp>("MyClass");
  cls.getBody().emplaceBlock();

  Evaluator evaluator(mod);

  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_EQ(diag.str(), "field \"foo\" does not exist");
  });

  auto result = evaluator.instantiate(builder.getStringAttr("MyClass"), {});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue = llvm::cast<evaluator::ObjectValue>(result.value().get())
                        ->getField(builder.getStringAttr("foo"));

  ASSERT_FALSE(succeeded(fieldValue));
}

/// Success scenarios.

TEST(EvaluatorTests, InstantiateObjectWithParamField) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  StringRef params[] = {"param"};
  StringRef fields[] = {"field"};
  Type types[] = {circt::om::OMIntegerType::get(&context)};
  ClassOp::buildSimpleClassOp(builder, loc, "MyClass", params, fields, types);

  Evaluator evaluator(mod);

  auto result = evaluator.instantiate(
      builder.getStringAttr("MyClass"),
      getEvaluatorValuesFromAttributes(
          &context, {circt::om::IntegerAttr::get(
                        &context, builder.getI32IntegerAttr(42))}));

  ASSERT_TRUE(succeeded(result));

  auto fieldValue = llvm::cast<evaluator::AttributeValue>(
                        llvm::cast<evaluator::ObjectValue>(result.value().get())
                            ->getField(builder.getStringAttr("field"))
                            .value()
                            .get())
                        ->getAs<circt::om::IntegerAttr>();

  ASSERT_TRUE(fieldValue);
  ASSERT_EQ(fieldValue.getValue().getValue(), 42);
}

TEST(EvaluatorTests, InstantiateObjectWithConstantField) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  auto cls = builder.create<ClassOp>("MyClass");
  auto &body = cls.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  auto constant = builder.create<ConstantOp>(
      circt::om::IntegerAttr::get(&context, builder.getI32IntegerAttr(42)));
  builder.create<ClassFieldOp>("field", constant);

  Evaluator evaluator(mod);

  auto result = evaluator.instantiate(builder.getStringAttr("MyClass"), {});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue = cast<evaluator::AttributeValue>(
                        llvm::cast<evaluator::ObjectValue>(result.value().get())
                            ->getField(builder.getStringAttr("field"))
                            .value()
                            .get())
                        ->getAs<circt::om::IntegerAttr>();
  ASSERT_TRUE(fieldValue);
  ASSERT_EQ(fieldValue.getValue().getValue(), 42);
}

TEST(EvaluatorTests, InstantiateObjectWithChildObject) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  StringRef params[] = {"param"};
  StringRef fields[] = {"field"};
  Type types[] = {circt::om::OMIntegerType::get(&context)};
  auto innerCls = ClassOp::buildSimpleClassOp(builder, loc, "MyInnerClass",
                                              params, fields, types);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  auto cls = builder.create<ClassOp>("MyClass", params);
  auto &body = cls.getBody().emplaceBlock();
  body.addArgument(circt::om::OMIntegerType::get(&context), cls.getLoc());
  builder.setInsertionPointToStart(&body);
  auto object = builder.create<ObjectOp>(innerCls, body.getArguments());
  builder.create<ClassFieldOp>("field", object);

  Evaluator evaluator(mod);

  auto result = evaluator.instantiate(
      builder.getStringAttr("MyClass"),
      {std::make_shared<evaluator::AttributeValue>(circt::om::IntegerAttr::get(
          &context, builder.getI32IntegerAttr(42)))});

  ASSERT_TRUE(succeeded(result));

  auto *fieldValue = llvm::cast<evaluator::ObjectValue>(
      llvm::cast<evaluator::ObjectValue>(result.value().get())
          ->getField(builder.getStringAttr("field"))
          .value()
          .get());

  ASSERT_TRUE(fieldValue);

  auto innerFieldValue =
      llvm::cast<evaluator::AttributeValue>(
          fieldValue->getField(builder.getStringAttr("field")).value().get())
          ->getAs<circt::om::IntegerAttr>();

  ASSERT_EQ(innerFieldValue.getValue().getValue(), 42);
}

TEST(EvaluatorTests, InstantiateObjectWithFieldAccess) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  StringRef params[] = {"param"};
  StringRef fields[] = {"field"};
  Type types[] = {circt::om::OMIntegerType::get(&context)};
  auto innerCls = ClassOp::buildSimpleClassOp(builder, loc, "MyInnerClass",
                                              params, fields, types);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  auto cls = builder.create<ClassOp>("MyClass", params);
  auto &body = cls.getBody().emplaceBlock();
  body.addArgument(circt::om::OMIntegerType::get(&context), cls.getLoc());
  builder.setInsertionPointToStart(&body);
  auto object = builder.create<ObjectOp>(innerCls, body.getArguments());
  auto field =
      builder.create<ObjectFieldOp>(builder.getI32Type(), object,
                                    builder.getArrayAttr(FlatSymbolRefAttr::get(
                                        builder.getStringAttr("field"))));
  builder.create<ClassFieldOp>("field", field);

  Evaluator evaluator(mod);

  auto result = evaluator.instantiate(
      builder.getStringAttr("MyClass"),
      {std::make_shared<evaluator::AttributeValue>(circt::om::IntegerAttr::get(
          &context, builder.getI32IntegerAttr(42)))});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue = llvm::cast<evaluator::AttributeValue>(
                        llvm::cast<evaluator::ObjectValue>(result.value().get())
                            ->getField(builder.getStringAttr("field"))
                            .value()
                            .get())
                        ->getAs<circt::om::IntegerAttr>();

  ASSERT_TRUE(fieldValue);
  ASSERT_EQ(fieldValue.getValue().getValue(), 42);
}

TEST(EvaluatorTests, InstantiateObjectWithChildObjectMemoized) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  auto innerCls = builder.create<ClassOp>("MyInnerClass");
  innerCls.getBody().emplaceBlock();

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  auto cls = builder.create<ClassOp>("MyClass");
  auto &body = cls.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  auto object = builder.create<ObjectOp>(innerCls, body.getArguments());
  builder.create<ClassFieldOp>("field1", object);
  builder.create<ClassFieldOp>("field2", object);

  Evaluator evaluator(mod);

  auto result = evaluator.instantiate(builder.getStringAttr("MyClass"), {});

  ASSERT_TRUE(succeeded(result));

  auto *field1Value = llvm::cast<evaluator::ObjectValue>(
      llvm::cast<evaluator::ObjectValue>(result.value().get())
          ->getField(builder.getStringAttr("field1"))
          .value()
          .get());

  auto *field2Value = llvm::cast<evaluator::ObjectValue>(
      llvm::cast<evaluator::ObjectValue>(result.value().get())
          ->getField(builder.getStringAttr("field2"))
          .value()
          .get());

  auto fieldNames =
      llvm::cast<evaluator::ObjectValue>(result.value().get())->getFieldNames();

  ASSERT_TRUE(fieldNames.size() == 2);
  StringRef fieldNamesTruth[] = {"field1", "field2"};
  for (auto fieldName : llvm::enumerate(fieldNames)) {
    auto str = llvm::dyn_cast_or_null<StringAttr>(fieldName.value());
    ASSERT_TRUE(str);
    ASSERT_EQ(str.getValue(), fieldNamesTruth[fieldName.index()]);
  }

  ASSERT_TRUE(field1Value);
  ASSERT_TRUE(field2Value);

  ASSERT_EQ(field1Value, field2Value);
}

TEST(EvaluatorTests, AnyCastObject) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  auto innerCls = builder.create<ClassOp>("MyInnerClass");
  innerCls.getBody().emplaceBlock();

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  auto cls = builder.create<ClassOp>("MyClass");
  auto &body = cls.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  auto object = builder.create<ObjectOp>(innerCls, body.getArguments());
  auto cast = builder.create<AnyCastOp>(object);
  builder.create<ClassFieldOp>("field", cast);

  Evaluator evaluator(mod);

  auto result = evaluator.instantiate(builder.getStringAttr("MyClass"), {});

  ASSERT_TRUE(succeeded(result));

  auto *fieldValue = llvm::cast<evaluator::ObjectValue>(
      llvm::cast<evaluator::ObjectValue>(result.value().get())
          ->getField(builder.getStringAttr("field"))
          .value()
          .get());

  ASSERT_TRUE(fieldValue);

  ASSERT_EQ(fieldValue->getClassOp(), innerCls);
}

TEST(EvaluatorTests, AnyCastParam) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  auto innerCls = ClassOp::buildSimpleClassOp(
      builder, builder.getLoc(), "MyInnerClass", {"param"}, {"field"},
      {AnyType::get(&context)});

  auto i64 = builder.getIntegerType(64);
  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  StringRef params[] = {"param"};
  auto cls = builder.create<ClassOp>("MyClass", params);
  auto &body = cls.getBody().emplaceBlock();
  body.addArguments({i64}, {builder.getLoc()});
  builder.setInsertionPointToStart(&body);
  auto cast = builder.create<AnyCastOp>(body.getArgument(0));
  SmallVector<Value> objectParams = {cast};
  auto object = builder.create<ObjectOp>(innerCls, objectParams);
  builder.create<ClassFieldOp>("field", object);

  Evaluator evaluator(mod);

  auto result =
      evaluator.instantiate(builder.getStringAttr("MyClass"),
                            getEvaluatorValuesFromAttributes(
                                &context, {builder.getIntegerAttr(i64, 42)}));

  ASSERT_TRUE(succeeded(result));

  auto *fieldValue = llvm::cast<evaluator::ObjectValue>(
      llvm::cast<evaluator::ObjectValue>(result.value().get())
          ->getField(builder.getStringAttr("field"))
          .value()
          .get());

  ASSERT_TRUE(fieldValue);

  auto *innerFieldValue = llvm::cast<evaluator::AttributeValue>(
      fieldValue->getField(builder.getStringAttr("field")).value().get());

  ASSERT_EQ(innerFieldValue->getAs<mlir::IntegerAttr>().getValue(), 42);
}

TEST(EvaluatorTests, InstantiateGraphRegion) {
  StringRef module =
      "!ty = !om.class.type<@LinkedList>"
      "om.class @LinkedList(%n: !ty, %val: !om.string) {"
      "  om.class.field @n, %n : !ty"
      "  om.class.field @val, %val : !om.string"
      "}"
      "om.class @ReferenceEachOther() {"
      "  %str = om.constant \"foo\" : !om.string"
      "  %val = om.object.field %1, [@n, @n, @val] : (!ty) -> !om.string"
      "  %0 = om.object @LinkedList(%1, %val) : (!ty, !om.string) -> !ty"
      "  %1 = om.object @LinkedList(%0, %str) : (!ty, !om.string) -> !ty"
      "  om.class.field @field1, %0 : !ty"
      "  om.class.field @field2, %1 : !ty"
      "}";

  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(module, ParserConfig(&context));

  Evaluator evaluator(owning.release());

  auto result = evaluator.instantiate(
      StringAttr::get(&context, "ReferenceEachOther"), {});

  ASSERT_TRUE(succeeded(result));

  auto *field1 = llvm::cast<evaluator::ObjectValue>(result.value().get())
                     ->getField("field1")
                     .value()
                     .get();
  auto *field2 = llvm::cast<evaluator::ObjectValue>(result.value().get())
                     ->getField("field2")
                     .value()
                     .get();

  ASSERT_EQ(
      field1,
      llvm::cast<evaluator::ObjectValue>(field2)->getField("n").value().get());
  ASSERT_EQ(
      field2,
      llvm::cast<evaluator::ObjectValue>(field1)->getField("n").value().get());

  ASSERT_EQ("foo", llvm::cast<evaluator::AttributeValue>(
                       llvm::cast<evaluator::ObjectValue>(field1)
                           ->getField("val")
                           .value()
                           .get())
                       ->getAs<StringAttr>()
                       .getValue());
}

TEST(EvaluatorTests, InstantiateCycle) {
  StringRef module = "!ty = !om.class.type<@LinkedList>"
                     "om.class @LinkedList(%n: !ty) {"
                     "  om.class.field @n, %n : !ty"
                     "}"
                     "om.class @ReferenceEachOther() {"
                     "  %val = om.object.field %0, [@n] : (!ty) -> !ty"
                     "  %0 = om.object @LinkedList(%val) : (!ty) -> !ty"
                     "  om.class.field @field, %0 : !ty"
                     "}";

  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_EQ(diag.str(), "failed to finalize evaluation. Probably the class "
                          "contains a dataflow cycle");
  });

  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(module, ParserConfig(&context));

  Evaluator evaluator(owning.release());

  auto result = evaluator.instantiate(
      StringAttr::get(&context, "ReferenceEachOther"), {});

  ASSERT_TRUE(failed(result));
}

} // namespace
