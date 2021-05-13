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

/// Given a string \p target, return a target with any subfield or subindex
/// references stripped.  Populate a \p annotations with any discovered
/// references.  This function assumes that the \p target string is a well
/// formed Annotation Target.
static StringRef parseSubFieldSubIndexAnnotations(StringRef target,
                                                  ArrayAttr &annotations,
                                                  MLIRContext *context) {
  if (target.empty())
    return target;

  auto fieldBegin = target.find_first_of(".[");

  // Exit if the target does not contain a subfield or subindex.
  if (fieldBegin == StringRef::npos)
    return target;

  auto targetBase = target.take_front(fieldBegin);
  target = target.substr(fieldBegin);
  SmallVector<Attribute> annotationVec;
  SmallString<16> temp;
  temp.push_back(target[0]);
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

  return targetBase;
}

/// Return an input \p target string in canonical form.  This converts a Legacy
/// Annotation (e.g., A.B.C) into a modern annotation (e.g., ~A|B>C).  Trailing
/// subfield/subindex references are preserved.
static std::string canonicalizeTarget(StringRef target) {

  // If this is a normal Target (not a Named), erase that field in the JSON
  // object and return that Target.
  if (target[0] == '~')
    return target.str();

  // This is a legacy target using the firrtl.annotations.Named type.  This
  // can be trivially canonicalized to a non-legacy target, so we do it with
  // the following three mappings:
  //   1. CircuitName => CircuitTarget, e.g., A -> ~A
  //   2. ModuleName => ModuleTarget, e.g., A.B -> ~A|B
  //   3. ComponentName => ReferenceTarget, e.g., A.B.C -> ~A|B>C
  std::string newTarget = "~";
  llvm::raw_string_ostream s(newTarget);
  bool isModule = true;
  for (auto a : target) {
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
  return newTarget;
}

/// Deserialize a JSON value into FIRRTL Annotations.  Annotations are
/// represented as a Target-keyed arrays of attributes.  The input JSON value is
/// checked, at runtime, to be an array of objects.  Returns true if successful,
/// false if unsuccessful.
bool circt::firrtl::fromJSON(json::Value &value,
                             llvm::StringMap<ArrayAttr> &annotationMap,
                             json::Path path, MLIRContext *context,
                             unsigned &annotationID) {

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
    auto maybeTarget = object->get("target");
    if (!maybeTarget)
      return llvm::Optional<std::string>("~");

    // Find the target.
    auto target = canonicalizeTarget(maybeTarget->getAsString().getValue());

    // If the target is something that we know we don't support, then error.
    bool unsupported = std::any_of(target.begin(), target.end(),
                                   [](char a) { return a == '/' || a == ':'; });
    if (unsupported) {
      p.field("target").report(
          "Unsupported target (not a local CircuitTarget, ModuleTarget, or "
          "ReferenceTarget without subfield or subindex)");
      return {};
    }

    // Remove the target field from the annotation and return the target.
    object->erase("target");
    return llvm::Optional<std::string>(target);
  };

  /// Convert arbitrary JSON to an MLIR Attribute.
  std::function<Attribute(json::Value &, json::Path)> convertJSONToAttribute =
      [&](json::Value &value, json::Path p) -> Attribute {
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
        return convertJSONToAttribute(unquotedValue.get(), p);
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

    // Build up the Attribute to represent the Annotation and store it in the
    // global Target -> Attribute mapping.
    NamedAttrList metadata;
    // Annotations on the element instance.
    ArrayAttr elementAnnotations;
    targetStrRef = parseSubFieldSubIndexAnnotations(
        targetStrRef, elementAnnotations, context);
    // Create an annotations with key "target", which will be parsed by
    // lowerTypes, and propagated to the appropriate instance.
    if (elementAnnotations && !elementAnnotations.empty())
      metadata.append("target", elementAnnotations);

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

  scatterCustomAnnotations(annotationMap, context, annotationID);

  return true;
}

/// Recursively walk a sifive.enterprise.grandcentral.AugmentedType to extract
/// any annotations it may contain.  This is going to generate two types of
/// annotations:
///   1) Annotations necessary to build interfaces and store them at "~"
///   2) Scattered annotations for how components bind to interfaces
static void parseAugmentedType(
    MLIRContext *context, DictionaryAttr augmentedType,
    llvm::StringMap<llvm::SmallVector<Attribute>> &newAnnotations,
    StringRef companion, StringAttr name, StringAttr defName) {

  /// Unpack a ReferenceTarget encoded as a DictionaryAttr.  Return a pair
  /// containing the Target string (up to the reference) and an array of
  /// components.  The input DicionaryAttr encoding is a JSON object of a
  /// serialized ReferenceTarget Scala class.  By example, this is
  /// converting:
  ///   ~Foo|Foo>a.b[0]
  /// To:
  ///   {"~Foo|Foo>a", {".b", "[0]"}}
  /// The format of a ReferenceTarget object like:
  ///   circuit: String
  ///   module: String
  ///   path: Seq[(Instance, OfModule)]
  ///   ref: String
  ///   component: Seq[TargetToken]
  auto refTargetToString =
      [&context](
          DictionaryAttr refTarget) -> std::pair<std::string, ArrayAttr> {
    auto circuit = refTarget.getAs<StringAttr>("circuit").getValue();
    auto module = refTarget.getAs<StringAttr>("module").getValue();
    auto path = refTarget.getAs<ArrayAttr>("path");
    if (!path.empty())
      llvm_unreachable("Unable to handle non-local annotations in GrandCentral "
                       "AugmentedTypes");
    auto ref = refTarget.getAs<StringAttr>("ref").getValue();
    SmallVector<Attribute> componentAttrs;
    for (auto component : refTarget.getAs<ArrayAttr>("component")) {
      auto dict = component.cast<DictionaryAttr>();
      auto clazz = dict.getAs<StringAttr>("class").getValue();
      auto value = dict.get("value");
      // A subfield like "bar" in "~Foo|Foo>foo.bar".
      if (auto field = value.dyn_cast<StringAttr>()) {
        assert(clazz == "firrtl.annotations.TargetToken$Field" &&
               "A StringAttr target token must be found with a subfield target "
               "token.");
        componentAttrs.push_back(
            StringAttr::get(context, (Twine(".") + field.getValue()).str()));
        continue;
      }
      // A subindex like "42" in "~Foo|Foo>foo[42]".
      if (auto index = value.dyn_cast<IntegerAttr>()) {
        assert(clazz == "firrtl.annotations.TargetToken$Index" &&
               "An IntegerAttr target token must be found with a subindex "
               "target token.");
        componentAttrs.push_back(StringAttr::get(
            context,
            (Twine("[") + index.getValue().toString(10, false) + "]").str()));
        continue;
      }
      // Any other token shouldn't exist.
      llvm_unreachable(
          "Unexpected TargetToken. Only Field or Index should be possible.");
    }
    return {(Twine("~" + circuit + "|" + module + ">" + ref)).str(),
            ArrayAttr::get(context, componentAttrs)};
  };

  /// The package name for all Grand Central Annotations
  static const char *package = "sifive.enterprise.grandcentral";

  auto classAttr = augmentedType.getAs<StringAttr>("class");

  // An AugmentedBundleType looks like:
  //   "defName": String
  //   "elements": Seq[AugmentedField]
  if (classAttr.getValue() == (Twine(package) + ".AugmentedBundleType").str()) {
    defName = augmentedType.getAs<StringAttr>("defName");

    // Each element is an AugmentedField with members:
    //   "name": String
    //   "description": Option[String]
    //   "tpe": AugmenetedType
    SmallVector<Attribute> elements;
    for (auto elt : augmentedType.getAs<ArrayAttr>("elements")) {
      auto field = elt.cast<DictionaryAttr>();
      auto name = field.getAs<StringAttr>("name");

      auto tpe = field.getAs<DictionaryAttr>("tpe");
      parseAugmentedType(context, tpe, newAnnotations, companion, name,
                         defName);

      // Collect information necessary to build a module with this view later.
      // This includes the optional description and name.
      NamedAttrList attrs;
      if (auto maybeDescription = field.get("description"))
        attrs.append("description", maybeDescription.cast<StringAttr>());
      attrs.append("name", name);
      elements.push_back(DictionaryAttr::getWithSorted(context, attrs));
    }
    // Add an annotation that stores information necessary to construct the
    // module for the view.  This needs the name of the module (defName) and the
    // names of the components inside it.
    NamedAttrList attrs;
    attrs.append("class", classAttr);
    attrs.append("defName", defName);
    attrs.append("elements", ArrayAttr::get(context, elements));
    newAnnotations["~"].push_back(
        DictionaryAttr::getWithSorted(context, attrs));
    return;
  }

  // An AugmentedGroundType looks like:
  //   "ref": ReferenceTarget
  //   "tpe": GroundType
  // The ReferenceTarget is not serialized to a string.  The GroundType will
  // either be an actual FIRRTL ground type or a GrandCentral uninferred type.
  // This can be ignored for us.
  if (classAttr.getValue() == (Twine(package) + ".AugmentedGroundType").str()) {
    auto target = refTargetToString(augmentedType.getAs<DictionaryAttr>("ref"));
    NamedAttrList attr;
    attr.append("class", classAttr);
    attr.append("defName", defName);
    attr.append("name", name);
    if (target.second)
      attr.append("target", target.second);
    newAnnotations[target.first].push_back(
        DictionaryAttr::getWithSorted(context, attr));
    return;
  }

  // An AugmentedVectorType looks like:
  //   "elements": Seq[AugmentedType]
  if (classAttr.getValue() == (Twine(package) + ".AugmentedVectorType").str()) {
    for (auto elt : augmentedType.getAs<ArrayAttr>("elements"))
      parseAugmentedType(context, elt.cast<DictionaryAttr>(), newAnnotations,
                         companion, name, defName);
    return;
  }

  llvm::errs() << "Handling of '" << classAttr << "' is not implemented yet!\n";
  llvm_unreachable("Code not implemented");
}

/// Convert known custom FIRRTL Annotations with compound targets to multiple
/// attributes that are attached to IR operations where they have semantic
/// meaning.  This rewrites the input \p annotationMap to convert non-specific
/// Annotations targeting "~" to those targeting something more specific if
/// possible.
void circt::firrtl::scatterCustomAnnotations(
    llvm::StringMap<ArrayAttr> &annotationMap, MLIRContext *context,
    unsigned &annotationID) {

  // Exit if not anotations exist that target "~".
  auto nonSpecificAnnotations = annotationMap["~"];
  if (!nonSpecificAnnotations)
    return;

  // Mutable store of new annotations produced.
  llvm::StringMap<llvm::SmallVector<Attribute>> newAnnotations;

  /// Return a new identifier that can be used to link scattered annotations
  /// together.  This mutates the by-reference parameter annotationID.
  auto newID = [&]() {
    return IntegerAttr::get(IntegerType::get(context, 64), annotationID++);
  };

  // Loop over all non-specific annotations that target "~".
  for (auto a : nonSpecificAnnotations) {
    auto dict = a.cast<DictionaryAttr>();
    StringAttr classAttr = dict.getAs<StringAttr>("class");
    // If the annotation doesn't have a "class" field, then we can't handle it.
    // Just copy it over.
    if (!classAttr) {
      newAnnotations["~"].push_back(a);
      continue;
    }

    // Get the "class" value and branch based on this.
    //
    // TODO: Determine a way to do this in an extensible way.  I.e., a user
    // should be able to register a handler for an annotation of a specific
    // class.
    StringRef clazz = classAttr.getValue();

    // Describes tap points into the design.  This has the following structure:
    //   blackBox: ModuleTarget
    //   keys: Seq[DataTapKey]
    // DataTapKey has multiple implementations:
    //   - ReferenceDataTapKey: (tapping a point which exists in the FIRRTL)
    //       portName: ReferenceTarget
    //       source: ReferenceTarget
    //   - DataTapModuleSignalKey: (tapping a point, by name, in a blackbox)
    //       portName: ReferenceTarget
    //       module: IsModule
    //       internalPath: String
    //   - DeletedDataTapKey: (not implemented here)
    //       portName: ReferenceTarget
    //   - LiteralDataTapKey: (not implemented here)
    //       portName: ReferenceTarget
    //       literal: Literal
    // A Literal is a FIRRTL IR literal serialized to a string.  For now, just
    // store the string.
    // TODO: Parse the literal string into a UInt or SInt literal.
    if (clazz == "sifive.enterprise.grandcentral.DataTapsAnnotation") {
      auto id = newID();
      NamedAttrList attrs;
      attrs.append("class", classAttr);
      auto target =
          canonicalizeTarget(dict.getAs<StringAttr>("blackBox").getValue());
      newAnnotations[target].push_back(
          DictionaryAttr::getWithSorted(context, attrs));

      // Process all the taps.
      for (auto b : dict.getAs<ArrayAttr>("keys")) {
        auto bDict = b.cast<DictionaryAttr>();
        auto classAttr = bDict.getAs<StringAttr>("class");

        // The "portName" field is common across all sub-types of DataTapKey.
        NamedAttrList port;
        auto portTarget =
            canonicalizeTarget(bDict.getAs<StringAttr>("portName").getValue());
        port.append("class", classAttr);
        port.append("id", id);

        if (classAttr.getValue() ==
            "sifive.enterprise.grandcentral.ReferenceDataTapKey") {
          NamedAttrList source;
          auto portID = newID();
          source.append("class", bDict.get("class"));
          source.append("id", id);
          source.append("portID", portID);
          newAnnotations[canonicalizeTarget(
                             bDict.getAs<StringAttr>("source").getValue())]
              .push_back(DictionaryAttr::getWithSorted(context, source));
          port.append("portID", portID);
          newAnnotations[portTarget].push_back(
              DictionaryAttr::getWithSorted(context, port));
          continue;
        }

        if (classAttr.getValue() ==
            "sifive.enterprise.grandcentral.DataTapModuleSignalKey") {
          NamedAttrList module;
          module.append("class", classAttr);
          module.append("id", id);
          module.append("internalPath",
                        bDict.getAs<StringAttr>("internalPath"));
          newAnnotations[canonicalizeTarget(
                             bDict.getAs<StringAttr>("module").getValue())]
              .push_back(DictionaryAttr::getWithSorted(context, module));
          newAnnotations[portTarget].push_back(
              DictionaryAttr::getWithSorted(context, port));
          continue;
        }

        if (classAttr.getValue() ==
            "sifive.enterprise.grandcentral.DeletedDataTapKey") {
          newAnnotations[portTarget].push_back(
              DictionaryAttr::getWithSorted(context, port));
          continue;
        }

        if (classAttr.getValue() ==
            "sifive.enterprise.grandcentral.LiteralDataTapKey") {
          NamedAttrList literal;
          literal.append("class", classAttr);
          literal.append("literal", bDict.getAs<StringAttr>("literal"));
          newAnnotations[canonicalizeTarget(
                             bDict.getAs<StringAttr>("portName").getValue())]
              .push_back(DictionaryAttr::getWithSorted(context, literal));
          continue;
        }

        llvm_unreachable("Unknown DataTapKey");
      }
      continue;
    }

    if (clazz == "sifive.enterprise.grandcentral.MemTapAnnotation") {
      auto id = newID();
      NamedAttrList attrs;
      auto target =
          canonicalizeTarget(dict.getAs<StringAttr>("source").getValue());
      attrs.append(dict.getNamed("class").getValue());
      attrs.append("id", id);
      newAnnotations[target].push_back(DictionaryAttr::get(context, attrs));
      for (auto b : dict.getAs<ArrayAttr>("taps")) {
        NamedAttrList foo;
        foo.append("class", dict.get("class"));
        foo.append("id", id);
        ArrayAttr subTargets;
        auto canonTarget = canonicalizeTarget(b.cast<StringAttr>().getValue());
        auto target =
            parseSubFieldSubIndexAnnotations(canonTarget, subTargets, context);
        if (subTargets && !subTargets.empty())
          foo.append("target", subTargets);
        newAnnotations[target].push_back(DictionaryAttr::get(context, foo));
      }
      continue;
    }

    if (clazz == "sifive.enterprise.grandcentral.GrandCentralView$"
                 "SerializedViewAnnotation") {
      auto id = newID();
      NamedAttrList companionAttrs, parentAttrs;
      companionAttrs.append("class", dict.get("class"));
      companionAttrs.append("id", id);
      companionAttrs.append("type", StringAttr::get(context, "companion"));
      auto companion = dict.getAs<StringAttr>("companion").getValue();
      newAnnotations[companion].push_back(
          DictionaryAttr::get(context, companionAttrs));
      auto parent = dict.getAs<StringAttr>("parent").getValue();
      parentAttrs.append("class", dict.get("class"));
      parentAttrs.append("id", id);
      parentAttrs.append("type", StringAttr::get(context, "parent"));
      newAnnotations[parent].push_back(
          DictionaryAttr::get(context, parentAttrs));
      auto view = dict.getAs<DictionaryAttr>("view");
      parseAugmentedType(context, view, newAnnotations, companion, {}, {});
      break;
    }

    if (clazz ==
        "sifive.enterprise.grandcentral.phases.SubCircuitsTargetDirectory") {
      llvm::errs() << "Handling for annotation '" << clazz
                   << "' not implemented\n";
      newAnnotations["~"].push_back(a);
      continue;
    }

    // Just copy over any annotation we don't understand.
    newAnnotations["~"].push_back(a);
  }

  // Delete all the old CircuitTarget annotations.
  annotationMap.erase("~");

  // Convert the mutable Annotation map to a SmallVector<ArrayAttr>.
  for (auto a : newAnnotations.keys()) {
    // If multiple annotations on a single object, then append it.
    if (annotationMap.count(a))
      for (auto attr : annotationMap[a])
        newAnnotations[a].push_back(attr);

    annotationMap[a] = ArrayAttr::get(context, newAnnotations[a]);
  }
}
