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
      [](llvm::json::Object *object) -> llvm::Optional<std::string> {
    // If a "targets" field exists, then this is likely a subclass of
    // firrtl.annotations.MultiTargetAnnotation.  We don't handle this yet, so
    // print an error and return none.
    if (object->get("targets")) {
      llvm::errs()
          << "Found a multitarget annotation. These are not yet supported.\n";
      return llvm::Optional<std::string>();
    }

    // If no "target" field exists, then exit, returning none.
    auto target = object->get("target");
    if (!target)
      return llvm::Optional<std::string>();

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

    // If the target is something that we know we don't support, then print an
    // error, promote the annotation to a Circuit annotation, and leave the
    // original target intact.  Otherwise, remove the target from the
    // annotation.
    bool unsupported =
        std::any_of(newTarget.begin(), newTarget.end(), [](char a) {
          return a == '/' || a == ':' || a == '>' || a == '.' || a == '[';
        });
    if (unsupported) {
      llvm::errs()
          << "Unsupported Annotation Target " << *target
          << ". (Only CircuitTarget and ModuleTarget are currently supported!) "
             "This will be promoted to a CircuitAnnotation.\n";
      newTarget = "~";
    } else {
      object->erase("target");
    }

    return llvm::Optional<std::string>(newTarget);
  };

  /// Convert arbitrary JSON to an MLIR Attribute.
  std::function<Attribute(llvm::json::Value &)> JSONToAttribute =
      [&](llvm::json::Value &value) -> Attribute {
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
      llvm::errs() << "Found a null in the JSON, returning 'nullptr'\n";
      return nullptr;
    }

    // Object
    if (auto a = value.getAsObject()) {
      NamedAttrList metadata;
      for (auto b : *a)
        metadata.append(b.first, JSONToAttribute(b.second));
      return DictionaryAttr::get(context, metadata);
    }

    // Array
    if (auto a = value.getAsArray()) {
      SmallVector<Attribute> metadata;
      for (auto b : *a)
        metadata.push_back(JSONToAttribute(b));
      return ArrayAttr::get(context, metadata);
    }

    llvm::errs() << "Unhandled JSON value: " << value << "\n";
    return nullptr;
  };

  // The JSON must be an object by definition of what an Annotation is.  Error
  // on anything else.
  auto object = value.getAsObject();
  if (!object) {
    path.report("Expected object");
    return false;
  }

  // Extra the "target" field from the Annotation object if it exists.  Remove
  // it so it isn't included in the Attribute.
  auto target = findAndEraseTarget(object);
  if (!target)
    target = llvm::Optional<std::string>("~");

  // Build up the Attribute to represent the Annotation and store it in the
  // global Target -> Attribute mapping.
  NamedAttrList metadata;
  for (auto field : *object)
    metadata.append(field.first, JSONToAttribute(field.second));
  annotationMap[target.getValue()].push_back(
      DictionaryAttr::get(context, metadata));
  return true;
};

} // namespace firrtl
} // namespace circt
