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

#include "circt/Dialect/FIRRTL/FIRAnnotations.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/JSON.h"

namespace json = llvm::json;

using namespace circt;
using namespace firrtl;

/// Given a string \p target, starting with either '.' or '[', this function
/// splits the string at every '[' and '.' and populates the \p annotations with
/// the array of strings. Assumption, the \p target string is a well formed
/// valid token specifying an instance of a bundle/array.
static void parseSubFieldSubIndexAnnotations(StringRef target,
                                             ArrayAttr &annotations,
                                             MLIRContext *context) {
  if (target.empty())
    return;
  char begin = target[0];
  SmallVector<Attribute> annotationVec;
  // The caller must strip the prefix, and the string target must only contain
  // the suffix.
  if (begin != '.' && begin != '[')
    return;
  SmallString<16> temp;
  temp.push_back(begin);
  for (size_t i = 1, s = target.size(); i < s; ++i) {
    if (target[i] == '[' || target[i] == '.') {
      // Create a StringAttr with the previous token.
      annotationVec.push_back(StringAttr::get(context, temp));
      temp.clear();
    }
    temp.push_back(target[i]);
  }
  // Save the last token.
  annotationVec.push_back(StringAttr::get(context, temp));
  annotations = ArrayAttr::get(context, annotationVec);
}

/// Deserialize a JSON value into FIRRTL Annotations.  Annotations are
/// represented as a Target-keyed arrays of attributes.  The input JSON value is
/// checked, at runtime, to be an array of objects.  Returns true if successful,
/// false if unsuccessful.
bool circt::firrtl::fromJSON(json::Value &value,
                             llvm::StringMap<ArrayAttr> &annotationMap,
                             json::Path path, MLIRContext *context) {

  /// Examine an Annotation JSON object and return an optional string indicating
  /// the target associated with this annotation.  Erase the target from the
  /// JSON object if a target was found.  Automatically convert any legacy Named
  /// targets to actual Targets.  Note: it is expected that a target may not
  /// exist, e.g., any subclass of firrtl.annotations.NoTargetAnnotation will
  /// not have a target.
  auto findAndEraseTarget = [](json::Object *object,
                               json::Path p) -> llvm::Optional<std::string> {
    // If no "target" field exists, then promote the annotation to a
    // CircuitTarget annotation by returning a target of "~".
    auto target = object->get("target");
    if (!target)
      return llvm::Optional<std::string>("~");

    // Find the target.
    auto maybeTarget = object->get("target")->getAsString();

    // If this is a normal Target (not a Named), erase that field in the JSON
    // object and return that Target.
    std::string newTarget;
    if (maybeTarget.getValue()[0] == '~') {
      newTarget = maybeTarget->str();
    } else {
      // This is a legacy target using the firrtl.annotations.Named type.  This
      // can be trivially canonicalized to a non-legacy target, so we do it with
      // the following three mappings:
      //   1. CircuitName => CircuitTarget, e.g., A -> ~A
      //   2. ModuleName => ModuleTarget, e.g., A.B -> ~A|B
      //   3. ComponentName => ReferenceTarget, e.g., A.B.C -> ~A|B>C
      newTarget = "~";
      llvm::raw_string_ostream s(newTarget);
      bool isModule = true;
      for (auto a : maybeTarget.getValue()) {
        switch (a) {
        case '.':
          if (isModule) {
            s << "|";
            isModule = false;
            break;
          }
          s << ">";
          break;
        default:
          s << a;
        }
      }
    }

    // If the target is something that we know we don't support, then error.
    bool unsupported = std::any_of(newTarget.begin(), newTarget.end(),
                                   [](char a) { return a == '/' || a == ':'; });
    if (unsupported) {
      p.field("target").report(
          "Unsupported target (not a local CircuitTarget, ModuleTarget, or "
          "ReferenceTarget without subfield or subindex)");
      return {};
    }

    // Remove the target field from the annotation and return the target.
    object->erase("target");
    return llvm::Optional<std::string>(newTarget);
  };

  /// Convert arbitrary JSON to an MLIR Attribute.
  std::function<Attribute(json::Value &, json::Path)> convertJSONToAttribute =
      [&](json::Value &value, json::Path p) -> Attribute {
    // String
    if (auto a = value.getAsString())
      return StringAttr::get(context, a.getValue());

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
        metadata.append(b.first,
                        convertJSONToAttribute(b.second, p.field(b.first)));
      return DictionaryAttr::get(context, metadata);
    }

    // Array
    if (auto a = value.getAsArray()) {
      SmallVector<Attribute> metadata;
      for (size_t i = 0, e = (*a).size(); i != e; ++i)
        metadata.push_back(convertJSONToAttribute((*a)[i], p.index(i)));
      return ArrayAttr::get(context, metadata);
    }

    llvm_unreachable("Impossible unhandled JSON type");
  };

  // The JSON value must be an array of objects.  Anything else is reported as
  // invalid.
  auto array = value.getAsArray();
  if (!array) {
    path.report(
        "Expected annotations to be an array, but found something else.");
    return false;
  }

  // Build a mutable map of Target to Annotation.
  llvm::StringMap<llvm::SmallVector<Attribute>> mutableAnnotationMap;
  for (size_t i = 0, e = (*array).size(); i != e; ++i) {
    auto object = (*array)[i].getAsObject();
    auto p = path.index(i);
    if (!object) {
      p.report("Expected annotations to be an array of objects, but found an "
               "array of something else.");
      return false;
    }
    // Find and remove the "target" field from the Annotation object if it
    // exists.  In the FIRRTL Dialect, the target will be implicitly specified
    // based on where the attribute is applied.
    auto optTarget = findAndEraseTarget(object, p);
    if (!optTarget)
      return false;
    StringRef targetStrRef = optTarget.getValue();
    // Get the position of the first '.' or '['.
    auto fieldBegin = targetStrRef.find_first_of(".[");

    // Build up the Attribute to represent the Annotation and store it in the
    // global Target -> Attribute mapping.
    NamedAttrList metadata;
    // Annotations on the element instance.
    ArrayAttr elementAnnotations;
    if (fieldBegin != StringRef::npos) {
      parseSubFieldSubIndexAnnotations(targetStrRef.substr(fieldBegin),
                                       elementAnnotations, context);
      // Create an annotations with key "target", which will be parsed by
      // lowerTypes, and propagated to the appropriate instance.
      metadata.append("target", elementAnnotations);
      targetStrRef = targetStrRef.substr(0, fieldBegin);
    }

    for (auto field : *object) {
      if (auto value = convertJSONToAttribute(field.second, p)) {
        metadata.append(field.first, value);
        continue;
      }
      return false;
    }
    mutableAnnotationMap[targetStrRef].push_back(
        DictionaryAttr::get(context, metadata));
  }

  // Convert the mutable Annotation map to a SmallVector<ArrayAttr>.
  for (auto a : mutableAnnotationMap.keys()) {
    // If multiple annotations on a single object, then append it.
    if (annotationMap.count(a))
      for (auto attr : annotationMap[a])
        mutableAnnotationMap[a].push_back(attr);

    annotationMap[a] = ArrayAttr::get(context, mutableAnnotationMap[a]);
  }

  return true;
}
