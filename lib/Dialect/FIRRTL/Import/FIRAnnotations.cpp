//===- FIRAnnotations.cpp - .fir to FIRRTL dialect parser -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provide utilities related to dealing with Scala FIRRTL Compiler Annotations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRAnnotations.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"

using namespace circt;
using namespace firrtl;

namespace circt {
namespace firrtl {

/// Deserialize a JSON value into a Target-keyed array of Annotations
/// (represented as DictionaryAttrs).
bool fromJSON(llvm::json::Value &value,
              llvm::StringMap<llvm::SmallVector<DictionaryAttr>> &annotationMap,
              llvm::json::Path path, MLIRContext *context) {

  /// Exampine an Annotation JSON object and return an optional string
  /// indicating the target associated with this annotation.  Erase the target
  /// from the JSON object if a target was found.  Automatically convert any
  /// legacy Named targets to actual Targets.  Note: it is expected that a
  /// target may not exist, e.g., any subclass of
  /// firrtl.annotations.NoTargetAnnotation will not have a target.
  auto findAndEraseTarget =
      [](llvm::json::Object *object,
         llvm::json::Path p) -> llvm::Optional<std::string> {
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
      // the following thre mappings:
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
    bool unsupported =
        std::any_of(newTarget.begin(), newTarget.end(), [](char a) {
          return a == '/' || a == ':' || a == '>' || a == '.' || a == '[';
        });
    if (unsupported) {
      p.field("target").report(
          "Unsupported target (not a CircuitTarget or ModuleTarget)");
      return {};
    }

    // Remove the target field from the annotation and return the target.
    object->erase("target");
    return llvm::Optional<std::string>(newTarget);
  };

  /// Convert arbitrary JSON to an MLIR Attribute.
  std::function<Attribute(llvm::json::Value &, llvm::json::Path)>
      JSONToAttribute =
          [&](llvm::json::Value &value, llvm::json::Path p) -> Attribute {
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
    if (auto a = value.getAsNull()) {
      p.report("Found a null in the JSON, returning 'nullptr'");
      return nullptr;
    }

    // Object
    if (auto a = value.getAsObject()) {
      NamedAttrList metadata;
      for (auto b : *a)
        metadata.append(b.first, JSONToAttribute(b.second, p.field(b.first)));
      return DictionaryAttr::get(context, metadata);
    }

    // Array
    if (auto a = value.getAsArray()) {
      SmallVector<Attribute> metadata;
      for (size_t i = 0, e = (*a).size(); i != e; ++i)
        metadata.push_back(JSONToAttribute((*a)[i], p.index(i)));
      return ArrayAttr::get(context, metadata);
    }

    llvm_unreachable("Impossible unhandled JSON type");
  };

  // The JSON value must be an array of objects.  Anything else is reported as
  // inavlid.
  auto array = value.getAsArray();
  if (!array) {
    path.report(
        "Expected annotations to be an array, but found something else.");
    return false;
  }

  for (size_t i = 0, e = (*array).size(); i != e; ++i) {
    auto object = (*array)[i].getAsObject();
    auto p = path.index(i);
    if (!object) {
      p.report("Expected annotations to be an array of objects, but found an "
               "array of something else.");
      return false;
    }
    // Extra the "target" field from the Annotation object if it exists.  Remove
    // it so it isn't included in the Attribute.
    auto target = findAndEraseTarget(object, p);
    if (!target)
      return false;

    // Build up the Attribute to represent the Annotation and store it in the
    // global Target -> Attribute mapping.
    NamedAttrList metadata;
    for (auto field : *object) {
      if (auto value = JSONToAttribute(field.second, p)) {
        metadata.append(field.first, value);
        continue;
      }
      return false;
    }
    annotationMap[target.getValue()].push_back(
        DictionaryAttr::get(context, metadata));
  }

  return true;
};

} // namespace firrtl
} // namespace circt
