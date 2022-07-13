//===- FIRAnnotations.cpp - FIRRTL Annotation Utilities -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provide utilities related to dealing with FIRRTL Annotations.
//
//===----------------------------------------------------------------------===//

#include "FIRAnnotations.h"

#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

namespace json = llvm::json;

using namespace circt;
using namespace firrtl;
using mlir::UnitAttr;

/// Convert arbitrary JSON to an MLIR Attribute.
static Attribute convertJSONToAttribute(MLIRContext *context,
                                        json::Value &value, json::Path p) {
  // String or quoted JSON
  if (auto a = value.getAsString()) {
    // Test to see if this might be quoted JSON (a string that is actually
    // JSON).  Sometimes FIRRTL developers will do this to serialize objects
    // that the Scala FIRRTL Compiler doesn't know about.
    auto unquotedValue = json::parse(a.getValue());
    auto err = unquotedValue.takeError();
    // If this parsed without an error, then it's more JSON and recurse on
    // that.
    if (!err)
      return convertJSONToAttribute(context, unquotedValue.get(), p);
    // If there was an error, then swallow it and handle this as a string.
    handleAllErrors(std::move(err), [&](const json::ParseError &a) {});
    return StringAttr::get(context, a.getValue());
  }

  // Integer
  if (auto a = value.getAsInteger())
    return IntegerAttr::get(IntegerType::get(context, 64), a.getValue());

  // Float
  if (auto a = value.getAsNumber())
    return FloatAttr::get(mlir::FloatType::getF64(context), a.getValue());

  // Boolean
  if (auto a = value.getAsBoolean())
    return BoolAttr::get(context, a.getValue());

  // Null
  if (auto a = value.getAsNull())
    return mlir::UnitAttr::get(context);

  // Object
  if (auto a = value.getAsObject()) {
    NamedAttrList metadata;
    for (auto b : *a)
      metadata.append(
          b.first, convertJSONToAttribute(context, b.second, p.field(b.first)));
    return DictionaryAttr::get(context, metadata);
  }

  // Array
  if (auto a = value.getAsArray()) {
    SmallVector<Attribute> metadata;
    for (size_t i = 0, e = (*a).size(); i != e; ++i)
      metadata.push_back(convertJSONToAttribute(context, (*a)[i], p.index(i)));
    return ArrayAttr::get(context, metadata);
  }

  llvm_unreachable("Impossible unhandled JSON type");
}

/// Convert a JSON value containing OMIR JSON (an array of OMNodes), convert
/// this to an OMIRAnnotation, and add it to a mutable `annotationMap` argument.
bool circt::firrtl::fromOMIRJSON(json::Value &value, StringRef circuitTarget,
                                 llvm::StringMap<ArrayAttr> &annotationMap,
                                 json::Path path, MLIRContext *context) {
  // The JSON value must be an array of objects.  Anything else is reported as
  // invalid.
  llvm::errs() << "start get array\n";
  auto *array = value.getAsArray();
  llvm::errs() << "stop get array\n";
  if (!array) {
    path.report(
        "Expected OMIR to be an array of nodes, but found something else.");
    return false;
  }

  SmallVector<Attribute> omnodes;
  auto parse = [&](size_t i) {
    auto *object = (*array)[i].getAsObject();
    auto p = path.index(i);
    if (!object) {
      p.report("Expected OMIR to be an array of objects, but found an array of "
               "something else.");
      return failure();
    }

    omnodes[i] = convertJSONToAttribute(context, (*array)[i], path);

    return success();
  };

  llvm::errs() << array->size() << "\n";
  omnodes.resize(array->size());
  NamedAttrList omirAnnoFields;
  omirAnnoFields.append("class", StringAttr::get(context, omirAnnoClass));
  llvm::errs() << "parallel start\n";
  if (failed(mlir::failableParallelForEachN(context, 0, array->size(), parse)))
    return false;
  llvm::errs() << "parallel start\n";
  omirAnnoFields.append("nodes", ArrayAttr::get(context, omnodes));

  DictionaryAttr omirAnno = DictionaryAttr::get(context, omirAnnoFields);

  // If no circuit annotations exist, just insert the OMIRAnnotation.
  auto &oldAnnotations = annotationMap[rawAnnotations];
  if (!oldAnnotations) {
    oldAnnotations = ArrayAttr::get(context, {omirAnno});
    return true;
  }

  // Rewrite the ArrayAttr for the circuit.
  SmallVector<Attribute> newAnnotations(oldAnnotations.begin(),
                                        oldAnnotations.end());
  newAnnotations.push_back(omirAnno);
  oldAnnotations = ArrayAttr::get(context, newAnnotations);

  return true;
}

/// Deserialize a JSON value into FIRRTL Annotations.  Annotations are
/// represented as a Target-keyed arrays of attributes.  The input JSON value is
/// checked, at runtime, to be an array of objects.  Returns true if successful,
/// false if unsuccessful.
bool circt::firrtl::fromJSONRaw(json::Value &value, StringRef circuitTarget,
                                SmallVectorImpl<Attribute> &attrs,
                                json::Path path, MLIRContext *context) {

  // The JSON value must be an array of objects.  Anything else is reported as
  // invalid.
  auto array = value.getAsArray();
  if (!array) {
    path.report(
        "Expected annotations to be an array, but found something else.");
    return false;
  }

  size_t offset = attrs.size();
  auto parse = [&](size_t i) {
    SmallVector<Attribute> foo;

    auto *object = (*array)[i].getAsObject();
    auto p = path.index(i);
    if (!object) {
      p.report("Expected annotations to be an array of objects, but found an "
               "array of something else.");
      return failure();
    }

    // Build up the Attribute to represent the Annotation
    NamedAttrList metadata;

    for (auto field : *object) {
      if (auto value = convertJSONToAttribute(context, field.second, p)) {
        metadata.append(field.first, value);
        continue;
      }
      return failure();
    }

    attrs[offset + i] = DictionaryAttr::get(context, metadata);
    return success();
  };

  attrs.resize(offset + array->size());
  if (failed(mlir::failableParallelForEachN(context, 0, array->size(), parse)))
    return false;

  return true;
}
