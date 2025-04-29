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
  auto &body = cls.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  builder.create<ClassFieldsOp>(loc, llvm::ArrayRef<mlir::Value>(),
                                ArrayAttr{});

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
  auto constantType = builder.getI32IntegerAttr(42);
  auto cls = builder.create<ClassOp>(
      "MyClass", builder.getStrArrayAttr({"field"}),
      builder.getDictionaryAttr({
          NamedAttribute(builder.getStringAttr("field"), constantType),

      }));
  auto &body = cls.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  auto constant = builder.create<ConstantOp>(
      circt::om::IntegerAttr::get(&context, constantType));
  builder.create<ClassFieldsOp>(loc, SmallVector<Value>({constant}),
                                ArrayAttr{});

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
  auto innerType = TypeAttr::get(ClassType::get(
      builder.getContext(), mlir::FlatSymbolRefAttr::get(innerCls)));
  auto cls = builder.create<ClassOp>(
      "MyClass", params, builder.getStrArrayAttr({"field"}),
      builder.getDictionaryAttr({
          NamedAttribute(builder.getStringAttr("field"), innerType),

      }));
  auto &body = cls.getBody().emplaceBlock();
  body.addArgument(circt::om::OMIntegerType::get(&context), cls.getLoc());
  builder.setInsertionPointToStart(&body);
  auto object = builder.create<ObjectOp>(innerCls, body.getArguments());
  builder.create<ClassFieldsOp>(loc, SmallVector<Value>({object}), ArrayAttr{});

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
  auto innerType = TypeAttr::get(ClassType::get(
      builder.getContext(), mlir::FlatSymbolRefAttr::get(innerCls)));
  auto cls = builder.create<ClassOp>(
      "MyClass", params, builder.getStrArrayAttr({"field"}),
      builder.getDictionaryAttr({
          NamedAttribute(builder.getStringAttr("field"), innerType),

      }));
  auto &body = cls.getBody().emplaceBlock();
  body.addArgument(circt::om::OMIntegerType::get(&context), cls.getLoc());
  builder.setInsertionPointToStart(&body);
  auto object = builder.create<ObjectOp>(innerCls, body.getArguments());
  auto field =
      builder.create<ObjectFieldOp>(builder.getI32Type(), object,
                                    builder.getArrayAttr(FlatSymbolRefAttr::get(
                                        builder.getStringAttr("field"))));
  builder.create<ClassFieldsOp>(loc, SmallVector<Value>({field}), ArrayAttr{});

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
  auto &innerBody = innerCls.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&innerBody);
  builder.create<ClassFieldsOp>(loc, llvm::ArrayRef<mlir::Value>(),
                                ArrayAttr{});

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  auto innerType = TypeAttr::get(ClassType::get(
      builder.getContext(), mlir::FlatSymbolRefAttr::get(innerCls)));
  auto cls = builder.create<ClassOp>(
      "MyClass", builder.getStrArrayAttr({"field1", "field2"}),
      builder.getDictionaryAttr({
          NamedAttribute(builder.getStringAttr("field1"), innerType),
          NamedAttribute(builder.getStringAttr("field2"), innerType),

      }));
  auto &body = cls.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  auto object = builder.create<ObjectOp>(innerCls, body.getArguments());
  builder.create<ClassFieldsOp>(loc, SmallVector<Value>({object, object}),
                                ArrayAttr{});

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
  auto &innerBody = innerCls.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&innerBody);
  builder.create<ClassFieldsOp>(loc, llvm::ArrayRef<mlir::Value>(),
                                ArrayAttr{});

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  auto innerType = TypeAttr::get(ClassType::get(
      builder.getContext(), mlir::FlatSymbolRefAttr::get(innerCls)));
  auto cls = builder.create<ClassOp>(
      "MyClass", builder.getStrArrayAttr({"field"}),
      builder.getDictionaryAttr({
          NamedAttribute(builder.getStringAttr("field"), innerType),

      }));
  auto &body = cls.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  auto object = builder.create<ObjectOp>(innerCls, body.getArguments());
  auto cast = builder.create<AnyCastOp>(object);
  builder.create<ClassFieldsOp>(loc, SmallVector<Value>({cast}), ArrayAttr{});

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
  auto innerType = TypeAttr::get(ClassType::get(
      builder.getContext(), mlir::FlatSymbolRefAttr::get(innerCls)));
  auto cls = builder.create<ClassOp>(
      "MyClass", params, builder.getStrArrayAttr({"field"}),
      builder.getDictionaryAttr({
          NamedAttribute(builder.getStringAttr("field"), innerType),

      }));
  auto &body = cls.getBody().emplaceBlock();
  body.addArguments({i64}, {builder.getLoc()});
  builder.setInsertionPointToStart(&body);
  auto cast = builder.create<AnyCastOp>(body.getArgument(0));
  SmallVector<Value> objectParams = {cast};
  auto object = builder.create<ObjectOp>(innerCls, objectParams);
  builder.create<ClassFieldsOp>(loc, SmallVector<Value>({object}), ArrayAttr{});

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
      "om.class @LinkedList(%n: !ty, %val: !om.string) -> (n: !ty, val: "
      "!om.string){"
      "  om.class.fields %n, %val : !ty, !om.string"
      "}"
      "om.class @ReferenceEachOther() -> (field1: !ty, field2: !ty) {"
      "  %str = om.constant \"foo\" : !om.string"
      "  %val = om.object.field %1, [@n, @n, @val] : (!ty) -> !om.string"
      "  %0 = om.object @LinkedList(%1, %val) : (!ty, !om.string) -> !ty"
      "  %1 = om.object @LinkedList(%0, %str) : (!ty, !om.string) -> !ty"
      "  om.class.fields %0, %1 : !ty, !ty"
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
                     "om.class @LinkedList(%n: !ty) -> (n: !ty){"
                     "  om.class.fields %n : !ty"
                     "}"
                     "om.class @ReferenceEachOther() -> (field: !ty){"
                     "  %val = om.object.field %0, [@n] : (!ty) -> !ty"
                     "  %0 = om.object @LinkedList(%val) : (!ty) -> !ty"
                     "  om.class.fields %0 : !ty"
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

TEST(EvaluatorTests, IntegerBinaryArithmeticAdd) {
  StringRef mod =
      "om.class @IntegerBinaryArithmeticAdd() -> (result: !om.integer) {"
      "  %0 = om.constant #om.integer<1 : si3> : !om.integer"
      "  %1 = om.constant #om.integer<2 : si3> : !om.integer"
      "  %2 = om.integer.add %0, %1 : !om.integer"
      "  om.class.fields %2 : !om.integer"
      "}";

  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(mod, ParserConfig(&context));

  Evaluator evaluator(owning.release());

  auto result = evaluator.instantiate(
      StringAttr::get(&context, "IntegerBinaryArithmeticAdd"), {});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue = llvm::cast<evaluator::ObjectValue>(result.value().get())
                        ->getField("result")
                        .value();

  ASSERT_EQ(3, llvm::cast<evaluator::AttributeValue>(fieldValue.get())
                   ->getAs<circt::om::IntegerAttr>()
                   .getValue()
                   .getValue());
}

TEST(EvaluatorTests, IntegerBinaryArithmeticMul) {
  StringRef mod =
      "om.class @IntegerBinaryArithmeticMul() -> (result: !om.integer) {"
      "  %0 = om.constant #om.integer<2 : si3> : !om.integer"
      "  %1 = om.constant #om.integer<3 : si3> : !om.integer"
      "  %2 = om.integer.mul %0, %1 : !om.integer"
      "  om.class.fields %2 : !om.integer"
      "}";

  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(mod, ParserConfig(&context));

  Evaluator evaluator(owning.release());

  auto result = evaluator.instantiate(
      StringAttr::get(&context, "IntegerBinaryArithmeticMul"), {});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue = llvm::cast<evaluator::ObjectValue>(result.value().get())
                        ->getField("result")
                        .value();

  ASSERT_EQ(6, llvm::cast<evaluator::AttributeValue>(fieldValue.get())
                   ->getAs<circt::om::IntegerAttr>()
                   .getValue()
                   .getValue());
}

TEST(EvaluatorTests, IntegerBinaryArithmeticShr) {
  StringRef mod =
      "om.class @IntegerBinaryArithmeticShr() -> (result: !om.integer){"
      "  %0 = om.constant #om.integer<8 : si5> : !om.integer"
      "  %1 = om.constant #om.integer<2 : si3> : !om.integer"
      "  %2 = om.integer.shr %0, %1 : !om.integer"
      "  om.class.fields %2 : !om.integer"
      "}";

  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(mod, ParserConfig(&context));

  Evaluator evaluator(owning.release());

  auto result = evaluator.instantiate(
      StringAttr::get(&context, "IntegerBinaryArithmeticShr"), {});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue = llvm::cast<evaluator::ObjectValue>(result.value().get())
                        ->getField("result")
                        .value();

  ASSERT_EQ(2, llvm::cast<evaluator::AttributeValue>(fieldValue.get())
                   ->getAs<circt::om::IntegerAttr>()
                   .getValue()
                   .getValue());
}

TEST(EvaluatorTests, IntegerBinaryArithmeticShrNegative) {
  StringRef mod =
      "om.class @IntegerBinaryArithmeticShrNegative() -> (result: !om.integer){"
      "  %0 = om.constant #om.integer<8 : si5> : !om.integer"
      "  %1 = om.constant #om.integer<-2 : si3> : !om.integer"
      "  %2 = om.integer.shr %0, %1 : !om.integer"
      "  om.class.fields %2 : !om.integer"
      "}";

  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    if (StringRef(diag.str()).starts_with("'om.integer.shr'"))
      ASSERT_EQ(diag.str(),
                "'om.integer.shr' op shift amount must be non-negative");
    if (StringRef(diag.str()).starts_with("failed"))
      ASSERT_EQ(diag.str(), "failed to evaluate integer operation");
  });

  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(mod, ParserConfig(&context));

  Evaluator evaluator(owning.release());

  auto result = evaluator.instantiate(
      StringAttr::get(&context, "IntegerBinaryArithmeticShrNegative"), {});

  ASSERT_TRUE(failed(result));
}

TEST(EvaluatorTests, IntegerBinaryArithmeticShrTooLarge) {
  StringRef mod =
      "om.class @IntegerBinaryArithmeticShrTooLarge() -> (result: !om.integer){"
      "  %0 = om.constant #om.integer<8 : si5> : !om.integer"
      "  %1 = om.constant #om.integer<36893488147419100000 : si66> "
      ": !om.integer"
      "  %2 = om.integer.shr %0, %1 : !om.integer"
      "  om.class.fields %2 : !om.integer"
      "}";

  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    if (StringRef(diag.str()).starts_with("'om.integer.shr'"))
      ASSERT_EQ(
          diag.str(),
          "'om.integer.shr' op shift amount must be representable in 64 bits");
    if (StringRef(diag.str()).starts_with("failed"))
      ASSERT_EQ(diag.str(), "failed to evaluate integer operation");
  });

  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(mod, ParserConfig(&context));

  Evaluator evaluator(owning.release());

  auto result = evaluator.instantiate(
      StringAttr::get(&context, "IntegerBinaryArithmeticShrTooLarge"), {});

  ASSERT_TRUE(failed(result));
}

TEST(EvaluatorTests, IntegerBinaryArithmeticShl) {
  StringRef mod =
      "om.class @IntegerBinaryArithmeticShl() -> (result: !om.integer){"
      "  %0 = om.constant #om.integer<8 : si7> : !om.integer"
      "  %1 = om.constant #om.integer<2 : si3> : !om.integer"
      "  %2 = om.integer.shl %0, %1 : !om.integer"
      "  om.class.fields %2 : !om.integer"
      "}";

  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(mod, ParserConfig(&context));

  Evaluator evaluator(owning.release());

  auto result = evaluator.instantiate(
      StringAttr::get(&context, "IntegerBinaryArithmeticShl"), {});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue = llvm::cast<evaluator::ObjectValue>(result.value().get())
                        ->getField("result")
                        .value();

  ASSERT_EQ(32, llvm::cast<evaluator::AttributeValue>(fieldValue.get())
                    ->getAs<circt::om::IntegerAttr>()
                    .getValue()
                    .getValue());
}

TEST(EvaluatorTests, IntegerBinaryArithmeticShlNegative) {
  StringRef mod = "om.class @IntegerBinaryArithmeticShlNegative() -> (result: "
                  "!om.integer) {"
                  "  %0 = om.constant #om.integer<8 : si5> : !om.integer"
                  "  %1 = om.constant #om.integer<-2 : si3> : !om.integer"
                  "  %2 = om.integer.shl %0, %1 : !om.integer"
                  "  om.class.fields %2 : !om.integer"
                  "}";

  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    if (StringRef(diag.str()).starts_with("'om.integer.shl'"))
      ASSERT_EQ(diag.str(),
                "'om.integer.shl' op shift amount must be non-negative");
    if (StringRef(diag.str()).starts_with("failed"))
      ASSERT_EQ(diag.str(), "failed to evaluate integer operation");
  });

  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(mod, ParserConfig(&context));

  Evaluator evaluator(owning.release());

  auto result = evaluator.instantiate(
      StringAttr::get(&context, "IntegerBinaryArithmeticShlNegative"), {});

  ASSERT_TRUE(failed(result));
}

TEST(EvaluatorTests, IntegerBinaryArithmeticShlTooLarge) {
  StringRef mod = "om.class @IntegerBinaryArithmeticShlTooLarge() -> (result: "
                  "!om.integer) {"
                  "  %0 = om.constant #om.integer<8 : si5> : !om.integer"
                  "  %1 = om.constant #om.integer<36893488147419100000 : si66> "
                  ": !om.integer"
                  "  %2 = om.integer.shl %0, %1 : !om.integer"
                  "  om.class.fields %2 : !om.integer"
                  "}";

  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    if (StringRef(diag.str()).starts_with("'om.integer.shl'"))
      ASSERT_EQ(
          diag.str(),
          "'om.integer.shl' op shift amount must be representable in 64 bits");
    if (StringRef(diag.str()).starts_with("failed"))
      ASSERT_EQ(diag.str(), "failed to evaluate integer operation");
  });

  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(mod, ParserConfig(&context));

  Evaluator evaluator(owning.release());

  auto result = evaluator.instantiate(
      StringAttr::get(&context, "IntegerBinaryArithmeticShlTooLarge"), {});

  ASSERT_TRUE(failed(result));
}

TEST(EvaluatorTests, IntegerBinaryArithmeticObjects) {
  StringRef mod =
      "om.class @Class1() -> (value: !om.integer){"
      "  %0 = om.constant #om.integer<1 : si3> : !om.integer"
      "  om.class.fields %0 : !om.integer"
      "}"
      ""
      "om.class @Class2() -> (value: !om.integer){"
      "  %0 = om.constant #om.integer<2 : si3> : !om.integer"
      "  om.class.fields %0 : !om.integer"
      "}"
      ""
      "om.class @IntegerBinaryArithmeticObjects() -> (result: !om.integer) {"
      "  %0 = om.object @Class1() : () -> !om.class.type<@Class1>"
      "  %1 = om.object.field %0, [@value] : "
      "(!om.class.type<@Class1>) -> !om.integer"
      ""
      "  %2 = om.object @Class2() : () -> !om.class.type<@Class2>"
      "  %3 = om.object.field %2, [@value] : "
      "(!om.class.type<@Class2>) -> !om.integer"
      ""
      "  %5 = om.integer.add %1, %3 : !om.integer"
      "  om.class.fields %5 : !om.integer"
      "}";

  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(mod, ParserConfig(&context));

  Evaluator evaluator(owning.release());

  auto result = evaluator.instantiate(
      StringAttr::get(&context, "IntegerBinaryArithmeticObjects"), {});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue = llvm::cast<evaluator::ObjectValue>(result.value().get())
                        ->getField("result")
                        .value();

  ASSERT_EQ(3, llvm::cast<evaluator::AttributeValue>(fieldValue.get())
                   ->getAs<circt::om::IntegerAttr>()
                   .getValue()
                   .getValue());
}

TEST(EvaluatorTests, IntegerBinaryArithmeticObjectsDelayed) {
  StringRef mod =
      "om.class @Class1(%input: !om.integer) -> (value: !om.integer, input: "
      "!om.integer) {"
      "  %0 = om.constant #om.integer<1 : si3> : !om.integer"
      "  om.class.fields %0, %input : !om.integer, !om.integer"
      "}"
      ""
      "om.class @Class2() -> (value: !om.integer){"
      "  %0 = om.constant #om.integer<2 : si3> : !om.integer"
      "  om.class.fields %0 : !om.integer"
      "}"
      ""
      "om.class @IntegerBinaryArithmeticObjectsDelayed() -> (result: "
      "!om.integer){"
      "  %0 = om.object @Class1(%5) : (!om.integer) -> !om.class.type<@Class1>"
      "  %1 = om.object.field %0, [@value] : "
      "(!om.class.type<@Class1>) -> !om.integer"
      ""
      "  %2 = om.object @Class2() : () -> !om.class.type<@Class2>"
      "  %3 = om.object.field %2, [@value] : "
      "(!om.class.type<@Class2>) -> !om.integer"
      ""
      "  %5 = om.integer.add %1, %3 : !om.integer"
      "  om.class.fields %5 : !om.integer"
      "}";

  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(mod, ParserConfig(&context));

  Evaluator evaluator(owning.release());

  auto result = evaluator.instantiate(
      StringAttr::get(&context, "IntegerBinaryArithmeticObjectsDelayed"), {});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue = llvm::cast<evaluator::ObjectValue>(result.value().get())
                        ->getField("result")
                        .value();

  ASSERT_EQ(3, llvm::cast<evaluator::AttributeValue>(fieldValue.get())
                   ->getAs<circt::om::IntegerAttr>()
                   .getValue()
                   .getValue());
}

TEST(EvaluatorTests, IntegerBinaryArithmeticWidthMismatch) {
  StringRef mod = "om.class @IntegerBinaryArithmeticWidthMismatch() -> "
                  "(result: !om.integer) {"
                  "  %0 = om.constant #om.integer<1 : si3> : !om.integer"
                  "  %1 = om.constant #om.integer<2 : si4> : !om.integer"
                  "  %2 = om.integer.add %0, %1 : !om.integer"
                  "  om.class.fields %2 : !om.integer"
                  "}";

  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(mod, ParserConfig(&context));

  Evaluator evaluator(owning.release());

  auto result = evaluator.instantiate(
      StringAttr::get(&context, "IntegerBinaryArithmeticWidthMismatch"), {});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue = llvm::cast<evaluator::ObjectValue>(result.value().get())
                        ->getField("result")
                        .value();

  ASSERT_EQ(3, llvm::cast<evaluator::AttributeValue>(fieldValue.get())
                   ->getAs<circt::om::IntegerAttr>()
                   .getValue()
                   .getValue());
}

TEST(EvaluatorTests, ListConcat) {
  StringRef mod = "om.class @ListConcat() -> (result: !om.list<!om.integer>) {"
                  "  %0 = om.constant #om.integer<0 : i8> : !om.integer"
                  "  %1 = om.constant #om.integer<1 : i8> : !om.integer"
                  "  %2 = om.constant #om.integer<2 : i8> : !om.integer"
                  "  %l0 = om.list_create %0, %1 : !om.integer"
                  "  %l1 = om.list_create %2 : !om.integer"
                  "  %concat = om.list_concat %l0, %l1 : !om.list<!om.integer>"
                  "  om.class.fields %concat : !om.list<!om.integer>"
                  "}";

  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(mod, ParserConfig(&context));

  Evaluator evaluator(owning.release());

  auto result =
      evaluator.instantiate(StringAttr::get(&context, "ListConcat"), {});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue = llvm::cast<evaluator::ObjectValue>(result.value().get())
                        ->getField("result")
                        .value();

  auto finalList =
      llvm::cast<evaluator::ListValue>(fieldValue.get())->getElements();

  ASSERT_EQ(3U, finalList.size());

  ASSERT_EQ(0, llvm::cast<evaluator::AttributeValue>(finalList[0].get())
                   ->getAs<circt::om::IntegerAttr>()
                   .getValue()
                   .getValue());

  ASSERT_EQ(1, llvm::cast<evaluator::AttributeValue>(finalList[1].get())
                   ->getAs<circt::om::IntegerAttr>()
                   .getValue()
                   .getValue());

  ASSERT_EQ(2, llvm::cast<evaluator::AttributeValue>(finalList[2].get())
                   ->getAs<circt::om::IntegerAttr>()
                   .getValue()
                   .getValue());
}

TEST(EvaluatorTests, ListConcatField) {
  StringRef mod =
      "om.class @ListField() -> (value: !om.list<!om.integer>) {"
      "  %0 = om.constant #om.integer<2 : i8> : !om.integer"
      "  %1 = om.list_create %0 : !om.integer"
      "  om.class.fields %1 : !om.list<!om.integer>"
      "}"
      "om.class @ListConcatField() -> (result: !om.list<!om.integer>){"
      "  %listField = om.object @ListField() : () -> !om.class.type<@ListField>"
      "  %0 = om.constant #om.integer<0 : i8> : !om.integer"
      "  %1 = om.constant #om.integer<1 : i8> : !om.integer"
      "  %l0 = om.list_create %0, %1 : !om.integer"
      "  %l1 = om.object.field %listField, [@value] : "
      "(!om.class.type<@ListField>) -> !om.list<!om.integer>"
      "  %concat = om.list_concat %l0, %l1 : !om.list<!om.integer>"
      "  om.class.fields %concat : !om.list<!om.integer>"
      "}";

  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(mod, ParserConfig(&context));

  Evaluator evaluator(owning.release());

  auto result =
      evaluator.instantiate(StringAttr::get(&context, "ListConcatField"), {});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue = llvm::cast<evaluator::ObjectValue>(result.value().get())
                        ->getField("result")
                        .value();

  auto finalList =
      llvm::cast<evaluator::ListValue>(fieldValue.get())->getElements();

  ASSERT_EQ(3U, finalList.size());

  ASSERT_EQ(0, llvm::cast<evaluator::AttributeValue>(finalList[0].get())
                   ->getAs<circt::om::IntegerAttr>()
                   .getValue()
                   .getValue());

  ASSERT_EQ(1, llvm::cast<evaluator::AttributeValue>(finalList[1].get())
                   ->getAs<circt::om::IntegerAttr>()
                   .getValue()
                   .getValue());

  ASSERT_EQ(2, llvm::cast<evaluator::AttributeValue>(finalList[2].get())
                   ->getAs<circt::om::IntegerAttr>()
                   .getValue()
                   .getValue());
}

TEST(EvaluatorTests, ListOfListConcat) {
  StringRef mod = "om.class @ListOfListConcat()  -> (result: "
                  "    !om.list<!om.list<!om.string>>) {"
                  "  %0 = om.constant \"foo\" : !om.string"
                  "  %1 = om.constant \"bar\" : !om.string"
                  "  %2 = om.constant \"baz\" : !om.string"
                  "  %3 = om.constant \"qux\" : !om.string"
                  "  %4 = om.list_create %0, %1 : !om.string"
                  "  %5 = om.list_create %4 : !om.list<!om.string>"
                  "  %6 = om.list_create %2, %3 : !om.string"
                  "  %7 = om.list_create %6 : !om.list<!om.string>"
                  "  %8 = om.list_concat %5, %7 : <!om.list<!om.string>>"
                  "  om.class.fields %8 : !om.list<!om.list<!om.string>> "
                  "}";

  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(mod, ParserConfig(&context));

  Evaluator evaluator(owning.release());

  auto result =
      evaluator.instantiate(StringAttr::get(&context, "ListOfListConcat"), {});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue = llvm::cast<evaluator::ObjectValue>(result.value().get())
                        ->getField("result")
                        .value();

  auto finalList =
      llvm::cast<evaluator::ListValue>(fieldValue.get())->getElements();

  ASSERT_EQ(2U, finalList.size());

  auto sublist0 =
      llvm::cast<evaluator::ListValue>(finalList[0].get())->getElements();

  ASSERT_EQ("foo", llvm::cast<evaluator::AttributeValue>(sublist0[0].get())
                       ->getAs<StringAttr>()
                       .getValue()
                       .str());

  ASSERT_EQ("bar", llvm::cast<evaluator::AttributeValue>(sublist0[1].get())
                       ->getAs<StringAttr>()
                       .getValue()
                       .str());

  auto sublist1 =
      llvm::cast<evaluator::ListValue>(finalList[1].get())->getElements();

  ASSERT_EQ("baz", llvm::cast<evaluator::AttributeValue>(sublist1[0].get())
                       ->getAs<StringAttr>()
                       .getValue()
                       .str());

  ASSERT_EQ("qux", llvm::cast<evaluator::AttributeValue>(sublist1[1].get())
                       ->getAs<StringAttr>()
                       .getValue()
                       .str());
}

TEST(EvaluatorTests, TupleGet) {
  StringRef mod = "om.class @Tuple() -> (val: !om.string) {"
                  "  %int = om.constant 1 : i1"
                  "  %str = om.constant \"foo\" : !om.string"
                  "  %tuple = om.tuple_create %int, %str  : i1, !om.string"
                  "  %val = om.tuple_get %tuple[1]  : tuple<i1, !om.string>"
                  "  om.class.fields %val : !om.string"
                  "}";

  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(mod, ParserConfig(&context));

  Evaluator evaluator(owning.release());

  auto result = evaluator.instantiate(StringAttr::get(&context, "Tuple"), {});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue = llvm::cast<evaluator::ObjectValue>(result.value().get())
                        ->getField("val")
                        .value();

  ASSERT_EQ("foo", llvm::cast<evaluator::AttributeValue>(fieldValue.get())
                       ->getAs<StringAttr>()
                       .getValue()
                       .str());
}

TEST(EvaluatorTests, NestedReferenceValue) {
  StringRef mod =
      "om.class @Empty() {"
      "  om.class.fields"
      "}"
      "om.class @Any() -> (object: !om.any, string: !om.any) {"
      "  %0 = om.object @Empty() : () -> !om.class.type<@Empty>"
      "  %1 = om.any_cast %0 : (!om.class.type<@Empty>) -> !om.any"
      "  %2 = om.constant \"foo\" : !om.string"
      "  %3 = om.any_cast %2 : (!om.string) -> !om.any"
      "  om.class.fields %1, %3 : !om.any, !om.any"
      "}"
      "om.class @InnerClass1(%anyListIn: !om.list<!om.any>)  -> (any_list1: "
      "!om.list<!om.any>) {"
      "  om.class.fields %anyListIn : !om.list<!om.any>"
      "}"
      "om.class @InnerClass2(%anyListIn: !om.list<!om.any>)  -> (any_list2: "
      "!om.list<!om.any>) {"
      "  om.class.fields %anyListIn : !om.list<!om.any>"
      "}"
      "om.class @OuterClass2()  -> (om: !om.class.type<@InnerClass2>) {"
      "  %0 = om.object @InnerClass2(%5) : (!om.list<!om.any>) -> "
      "!om.class.type<@InnerClass2>"
      "  %1 = om.object @Any() : () -> !om.class.type<@Any>"
      "  %2 = om.object.field %1, [@object] : (!om.class.type<@Any>) -> "
      "!om.any"
      "  %3 = om.object @Any() : () -> !om.class.type<@Any>"
      "  %4 = om.object.field %3, [@object] : (!om.class.type<@Any>) -> "
      "!om.any"
      "  %5 = om.list_create %2, %4 : !om.any"
      "  om.class.fields %0 : !om.class.type<@InnerClass2>"
      "}"
      "om.class @OuterClass1()  -> (om: !om.any) {"
      "  %0 = om.object @InnerClass1(%8) : (!om.list<!om.any>) -> "
      "!om.class.type<@InnerClass1>"
      "  %1 = om.any_cast %0 : (!om.class.type<@InnerClass1>) -> !om.any"
      "  %2 = om.object @OuterClass2() : () -> !om.class.type<@OuterClass2>"
      "  %3 = om.object.field %2, [@om] : (!om.class.type<@OuterClass2>) -> "
      "!om.class.type<@InnerClass2>"
      "  %4 = om.any_cast %3 : (!om.class.type<@InnerClass2>) -> !om.any"
      "  %5 = om.object @OuterClass2() : () -> !om.class.type<@OuterClass2>"
      "  %6 = om.object.field %5, [@om] : (!om.class.type<@OuterClass2>) -> "
      "!om.class.type<@InnerClass2>"
      "  %7 = om.any_cast %6 : (!om.class.type<@InnerClass2>) -> !om.any"
      "  %8 = om.list_create %4, %7 : !om.any"
      "  om.class.fields %1 : !om.any"
      "}";

  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  OwningOpRef<ModuleOp> owning =
      parseSourceString<ModuleOp>(mod, ParserConfig(&context));

  Evaluator evaluator(owning.release());

  auto result =
      evaluator.instantiate(StringAttr::get(&context, "OuterClass1"), {});

  ASSERT_TRUE(succeeded(result));

  ASSERT_TRUE(isa<evaluator::ObjectValue>(
      llvm::cast<evaluator::ListValue>(
          llvm::cast<evaluator::ObjectValue>(
              llvm::cast<evaluator::ListValue>(
                  llvm::cast<evaluator::ObjectValue>(
                      llvm::cast<evaluator::ObjectValue>(result->get())
                          ->getField("om")
                          ->get())
                      ->getField("any_list1")
                      ->get())
                  ->getElements()[0]
                  .get())
              ->getField("any_list2")
              ->get())
          ->getElements()[0]
          .get()));
}

} // namespace
