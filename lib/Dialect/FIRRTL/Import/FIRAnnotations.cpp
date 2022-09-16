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
    auto unquotedValue = json::parse(*a);
    auto err = unquotedValue.takeError();
    // If this parsed without an error and we didn't just unquote a number, then
    // it's more JSON and recurse on that.
    //
    // We intentionally do not want to unquote a number as, in JSON, the string
    // "0" is different from the number 0.  If we conflate these, then later
    // expectations about annotation structure may be broken.  I.e., an
    // annotation expecting a string may see a number.
    if (!err && !unquotedValue.get().getAsNumber())
      return convertJSONToAttribute(context, unquotedValue.get(), p);
    // If there was an error, then swallow it and handle this as a string.
    handleAllErrors(std::move(err), [&](const json::ParseError &a) {});
    return StringAttr::get(context, *a);
  }

  // Integer
  if (auto a = value.getAsInteger())
    return IntegerAttr::get(IntegerType::get(context, 64), *a);

  // Float
  if (auto a = value.getAsNumber())
    return FloatAttr::get(mlir::FloatType::getF64(context), *a);

  // Boolean
  if (auto a = value.getAsBoolean())
    return BoolAttr::get(context, *a);

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
                                 SmallVectorImpl<Attribute> &annotations,
                                 json::Path path, MLIRContext *context) {
  // The JSON value must be an array of objects.  Anything else is reported as
  // invalid.
  auto *array = value.getAsArray();
  if (!array) {
    path.report(
        "Expected OMIR to be an array of nodes, but found something else.");
    return false;
  }

  // Build a mutable map of Target to Annotation.
  SmallVector<Attribute> omnodes;
  for (size_t i = 0, e = (*array).size(); i != e; ++i) {
    auto *object = (*array)[i].getAsObject();
    auto p = path.index(i);
    if (!object) {
      p.report("Expected OMIR to be an array of objects, but found an array of "
               "something else.");
      return false;
    }

    // Manually built up OMNode.
    NamedAttrList omnode;

    // Validate that this looks like an OMNode.  This should have three fields:
    //   - "info": String
    //   - "id": String that starts with "OMID:"
    //   - "fields": Array<Object>
    // Fields is optional and is a dictionary encoded as an array of objects:
    //   - "info": String
    //   - "name": String
    //   - "value": JSON
    // The dictionary is keyed by the "name" member and the array of fields is
    // guaranteed to not have collisions of the "name" key.
    auto maybeInfo = object->getString("info");
    if (!maybeInfo) {
      p.report("OMNode missing mandatory member \"info\" with type \"string\"");
      return false;
    }
    auto maybeID = object->getString("id");
    if (!maybeID || !maybeID->startswith("OMID:")) {
      p.report("OMNode missing mandatory member \"id\" with type \"string\" "
               "that starts with \"OMID:\"");
      return false;
    }
    auto *maybeFields = object->get("fields");
    if (maybeFields && !maybeFields->getAsArray()) {
      p.report("OMNode has \"fields\" member with incorrect type (expected "
               "\"array\")");
      return false;
    }
    Attribute fields;
    if (!maybeFields)
      fields = DictionaryAttr::get(context, {});
    else {
      auto array = *maybeFields->getAsArray();
      NamedAttrList fieldAttrs;
      for (size_t i = 0, e = array.size(); i != e; ++i) {
        auto *field = array[i].getAsObject();
        auto pI = p.field("fields").index(i);
        if (!field) {
          pI.report("OMNode has field that is not an \"object\"");
          return false;
        }
        auto maybeInfo = field->getString("info");
        if (!maybeInfo) {
          pI.report(
              "OMField missing mandatory member \"info\" with type \"string\"");
          return false;
        }
        auto maybeName = field->getString("name");
        if (!maybeName) {
          pI.report(
              "OMField missing mandatory member \"name\" with type \"string\"");
          return false;
        }
        auto *maybeValue = field->get("value");
        if (!maybeValue) {
          pI.report("OMField missing mandatory member \"value\"");
          return false;
        }
        NamedAttrList values;
        values.append("info", StringAttr::get(context, *maybeInfo));
        values.append("value", convertJSONToAttribute(context, *maybeValue,
                                                      pI.field("value")));
        fieldAttrs.append(*maybeName, DictionaryAttr::get(context, values));
      }
      fields = DictionaryAttr::get(context, fieldAttrs);
    }

    omnode.append("info", StringAttr::get(context, *maybeInfo));
    omnode.append("id", convertJSONToAttribute(context, *object->get("id"),
                                               p.field("id")));
    omnode.append("fields", fields);
    omnodes.push_back(DictionaryAttr::get(context, omnode));
  }

  NamedAttrList omirAnnoFields;
  omirAnnoFields.append("class", StringAttr::get(context, omirAnnoClass));
  omirAnnoFields.append("nodes", convertJSONToAttribute(context, value, path));

  DictionaryAttr omirAnno = DictionaryAttr::get(context, omirAnnoFields);
  annotations.push_back(omirAnno);

  return true;
}

/// Deserialize a JSON value into FIRRTL Annotations.  Annotations are
/// represented as a Target-keyed arrays of attributes.  The input JSON value is
/// checked, at runtime, to be an array of objects.  Returns true if successful,
/// false if unsuccessful.
bool circt::firrtl::fromJSONRaw(json::Value &value, StringRef circuitTarget,
                                SmallVectorImpl<Attribute> &annotations,
                                json::Path path, MLIRContext *context) {

  // The JSON value must be an array of objects.  Anything else is reported as
  // invalid.
  auto array = value.getAsArray();
  if (!array) {
    path.report(
        "Expected annotations to be an array, but found something else.");
    return false;
  }

  // Build an array of annotations.
  for (size_t i = 0, e = (*array).size(); i != e; ++i) {
    auto object = (*array)[i].getAsObject();
    auto p = path.index(i);
    if (!object) {
      p.report("Expected annotations to be an array of objects, but found an "
               "array of something else.");
      return false;
    }

    // Build up the Attribute to represent the Annotation
    NamedAttrList metadata;

    for (auto field : *object) {
      if (auto value = convertJSONToAttribute(context, field.second, p)) {
        metadata.append(field.first, value);
        continue;
      }
      return false;
    }

    annotations.push_back(DictionaryAttr::get(context, metadata));
  }

  return true;
}
