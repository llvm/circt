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

#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/JSON.h"

namespace json = llvm::json;

using namespace circt;
using namespace firrtl;

static constexpr const char *omirAnnotationClass =
    "freechips.rocketchip.objectmodel.OMIRAnnotation";

/// Split a target into a base target (including a reference if one exists) and
/// an optional array of subfield/subindex tokens.
static std::pair<StringRef, llvm::Optional<ArrayAttr>>
splitTarget(StringRef target, MLIRContext *context) {
  if (target.empty())
    return {target, None};

  // Find the string index where the target can be partitioned into the "base
  // target" and the "target".  The "base target" is the module or instance and
  // the "target" is everything else.  This has two variants that need to be
  // considered:
  //
  //   1) A Local target, e.g., ~Foo|Foo>bar.baz
  //   2) An instance target, e.g., ~Foo|Foo/bar:Bar>baz.qux
  //
  // In (1), this should be partitioned into ["~Foo|Foo>bar", ".baz"].  In (2),
  // this should be partitioned into ["~Foo|Foo/bar:Bar", ">baz.qux"].
  bool isInstance = false;
  size_t fieldBegin = target.find_if_not([&isInstance](char c) {
    switch (c) {
    case '/':
      return isInstance = true;
    case '>':
      return !isInstance;
    case '[':
    case '.':
      return false;
    default:
      return true;
    };
  });

  // Exit if the target does not contain a subfield or subindex.
  if (fieldBegin == StringRef::npos)
    return {target, None};

  auto targetBase = target.take_front(fieldBegin);
  target = target.substr(fieldBegin);
  SmallVector<Attribute> annotationVec;
  SmallString<16> temp;
  for (auto c : target.drop_front()) {
    if (c == ']') {
      // Create a IntegerAttr with the previous sub-index token.
      APInt subIndex;
      if (!temp.str().getAsInteger(10, subIndex))
        annotationVec.push_back(IntegerAttr::get(IntegerType::get(context, 64),
                                                 subIndex.getZExtValue()));
      else
        // We don't have a good way to emit error here. This will be reported as
        // an error in the FIRRTL parser.
        annotationVec.push_back(StringAttr::get(context, temp));
      temp.clear();
    } else if (c == '[' || c == '.') {
      // Create a StringAttr with the previous token.
      if (!temp.empty())
        annotationVec.push_back(StringAttr::get(context, temp));
      temp.clear();
    } else
      temp.push_back(c);
  }
  // Save the last token.
  if (!temp.empty())
    annotationVec.push_back(StringAttr::get(context, temp));

  return {targetBase, ArrayAttr::get(context, annotationVec)};
}

/// Split out non-local paths.  This will return a set of target strings for
/// each named entity along the path.
/// c|c:ai/Am:bi/Bm>d.agg[3] ->
/// c|c>ai, c|Am>bi, c|Bm>d.agg[2]
static SmallVector<std::tuple<std::string, std::string, std::string>>
expandNonLocal(StringRef target) {
  SmallVector<std::tuple<std::string, std::string, std::string>> retval;
  StringRef circuit;
  std::tie(circuit, target) = target.split('|');
  while (target.count(':')) {
    StringRef nla;
    std::tie(nla, target) = target.split(':');
    StringRef inst, mod;
    std::tie(mod, inst) = nla.split('/');
    retval.emplace_back((circuit + "|" + mod + ">" + inst).str(), mod.str(),
                        inst.str());
  }
  if (target.empty()) {
    retval.emplace_back(circuit.str(), "", "");
  } else {

    StringRef mod, name;
    // remove aggregate
    auto targetBase =
        target.take_until([](char c) { return c == '.' || c == '['; });
    std::tie(mod, name) = targetBase.split('>');
    if (name.empty())
      name = mod;
    retval.emplace_back((circuit + "|" + target).str(), mod, name);
  }
  return retval;
}

/// Make an anchor for a non-local annotation.  Use the expanded path to build
/// the module and name list in the anchor.
static FlatSymbolRefAttr buildNLA(
    CircuitOp circuit, size_t nlaSuffix,
    SmallVectorImpl<std::tuple<std::string, std::string, std::string>> &nlas) {
  OpBuilder b(circuit.getBodyRegion());
  SmallVector<Attribute> mods;
  SmallVector<Attribute> insts;
  for (auto &nla : nlas) {
    mods.push_back(
        FlatSymbolRefAttr::get(circuit.getContext(), std::get<1>(nla)));
    insts.push_back(StringAttr::get(circuit.getContext(), std::get<2>(nla)));
  }
  auto modAttr = ArrayAttr::get(circuit.getContext(), mods);
  auto instAttr = ArrayAttr::get(circuit.getContext(), insts);
  auto nla = b.create<NonLocalAnchor>(
      circuit.getLoc(), "nla_" + std::to_string(nlaSuffix), modAttr, instAttr);
  return FlatSymbolRefAttr::get(nla);
}

/// Append the argument `target` to the `annotation` using the key "target".
static inline void appendTarget(NamedAttrList &annotation, ArrayAttr target) {
  annotation.append("target", target);
}

/// Mutably update a prototype Annotation (stored as a `NamedAttrList`) with
/// subfield/subindex information from a Target string.  Subfield/subindex
/// information will be placed in the key "target" at the back of the
/// Annotation.  If no subfield/subindex information, the Annotation is
/// unmodified.  Return the split input target as a base target (include a
/// reference if one exists) and an optional array containing subfield/subindex
/// tokens.
static std::pair<StringRef, llvm::Optional<ArrayAttr>>
splitAndAppendTarget(NamedAttrList &annotation, StringRef target,
                     MLIRContext *context) {
  auto targetPair = splitTarget(target, context);
  if (targetPair.second.hasValue())
    appendTarget(annotation, targetPair.second.getValue());

  return targetPair;
}

/// Return an input \p target string in canonical form.  This converts a Legacy
/// Annotation (e.g., A.B.C) into a modern annotation (e.g., ~A|B>C).  Trailing
/// subfield/subindex references are preserved.
static llvm::Optional<std::string> canonicalizeTarget(StringRef target) {

  if (target.empty())
    return {};

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
  unsigned tokenIdx = 0;
  for (auto a : target) {
    if (a == '.') {
      switch (tokenIdx) {
      case 0:
        s << "|";
        break;
      case 1:
        s << ">";
        break;
      default:
        s << ".";
        break;
      }
      ++tokenIdx;
    } else
      s << a;
  }
  return llvm::Optional<std::string>(newTarget);
}

/// Implements the same behavior as DictionaryAttr::getAs<A> to return the value
/// of a specific type associated with a key in a dictionary.  However, this is
/// specialized to print a useful error message, specific to custom annotation
/// process, on failure.
template <typename A>
static A tryGetAs(DictionaryAttr &dict, const Attribute &root, StringRef key,
                  Location loc, Twine className, Twine path = Twine()) {
  // Check that the key exists.
  auto value = dict.get(key);
  if (!value) {
    SmallString<128> msg;
    if (path.isTriviallyEmpty())
      msg = ("Annotation '" + className + "' did not contain required key '" +
             key + "'.")
                .str();
    else
      msg = ("Annotation '" + className + "' with path '" + path +
             "' did not contain required key '" + key + "'.")
                .str();
    mlir::emitError(loc, msg).attachNote()
        << "The full Annotation is reproduced here: " << root << "\n";
    return nullptr;
  }
  // Check that the value has the correct type.
  auto valueA = value.dyn_cast_or_null<A>();
  if (!valueA) {
    SmallString<128> msg;
    if (path.isTriviallyEmpty())
      msg = ("Annotation '" + className +
             "' did not contain the correct type for key '" + key + "'.")
                .str();
    else
      msg = ("Annotation '" + className + "' with path '" + path +
             "' did not contain the correct type for key '" + key + "'.")
                .str();
    mlir::emitError(loc, msg).attachNote()
        << "The full Annotation is reproduced here: " << root << "\n";
    return nullptr;
  }
  return valueA;
}

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

/// Deserialize a JSON value into FIRRTL Annotations.  Annotations are
/// represented as a Target-keyed arrays of attributes.  The input JSON value is
/// checked, at runtime, to be an array of objects.  Returns true if successful,
/// false if unsuccessful.
bool circt::firrtl::fromJSON(json::Value &value, StringRef circuitTarget,
                             llvm::StringMap<ArrayAttr> &annotationMap,
                             json::Path path, CircuitOp circuit,
                             size_t &nlaNumber) {
  auto context = circuit.getContext();

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
    auto maybeTargetStr = maybeTarget->getAsString();
    if (!maybeTargetStr) {
      p.field("target").report("target must be a string type");
      return {};
    }
    auto canonTargetStr = canonicalizeTarget(maybeTargetStr.getValue());
    if (!canonTargetStr) {
      p.field("target").report("invalid target string");
      return {};
    }

    auto target = canonTargetStr.getValue();

    // Remove the target field from the annotation and return the target.
    object->erase("target");
    return llvm::Optional<std::string>(target);
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

    if (targetStrRef != "~") {
      auto circuitFieldEnd = targetStrRef.find_first_of('|');
      if (circuitTarget != targetStrRef.take_front(circuitFieldEnd)) {
        p.report("annotation has invalid circuit name");
        return false;
      }
    }

    // Build up the Attribute to represent the Annotation and store it in the
    // global Target -> Attribute mapping.
    auto NLATargets = expandNonLocal(targetStrRef);

    NamedAttrList metadata;
    // Annotations on the element instance.
    auto leafTarget =
        splitAndAppendTarget(metadata, std::get<0>(NLATargets.back()), context)
            .first;
    for (auto field : *object) {
      if (auto value = convertJSONToAttribute(context, field.second, p)) {
        metadata.append(field.first, value);
        continue;
      }
      return false;
    }
    FlatSymbolRefAttr nlaSym;
    if (NLATargets.size() > 1) {
      nlaSym = buildNLA(circuit, ++nlaNumber, NLATargets);
      metadata.append("circt.nonlocal", nlaSym);
    }
    mutableAnnotationMap[leafTarget].push_back(
        DictionaryAttr::get(context, metadata));

    for (int i = 0, e = NLATargets.size() - 1; i < e; ++i) {
      NamedAttrList pathmetadata;
      pathmetadata.append("circt.nonlocal", nlaSym);
      pathmetadata.append("class", StringAttr::get(context, "circt.nonlocal"));
      mutableAnnotationMap[std::get<0>(NLATargets[i])].push_back(
          DictionaryAttr::get(context, pathmetadata));
    }
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

bool circt::firrtl::fromOMIRJSON(json::Value &value, StringRef circuitTarget,
                                 llvm::StringMap<ArrayAttr> &annotationMap,
                                 json::Path path, CircuitOp circuit) {
  auto *context = circuit.getContext();

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
    if (!maybeID || !maybeID.getValue().startswith("OMID:")) {
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
        values.append("info", StringAttr::get(context, maybeInfo.getValue()));
        values.append("value", convertJSONToAttribute(context, *maybeValue,
                                                      pI.field("value")));
        fieldAttrs.append(maybeName.getValue(),
                          DictionaryAttr::get(context, values));
      }
      fields = DictionaryAttr::get(context, fieldAttrs);
    }

    omnode.append("info", StringAttr::get(context, maybeInfo.getValue()));
    omnode.append("id", convertJSONToAttribute(context, *object->get("id"),
                                               p.field("id")));
    omnode.append("fields", fields);
    omnodes.push_back(DictionaryAttr::get(context, omnode));
  }

  NamedAttrList omirAnnoFields;
  omirAnnoFields.append("class", StringAttr::get(context, omirAnnotationClass));
  omirAnnoFields.append("nodes", convertJSONToAttribute(context, value, path));

  DictionaryAttr omirAnno = DictionaryAttr::get(context, omirAnnoFields);

  // If no circuit annotations exist, just insert the OMIRAnnotation.
  auto &oldAnnotations = annotationMap["~"];
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

/// Recursively walk a sifive.enterprise.grandcentral.AugmentedType to extract
/// any annotations it may contain.  This is going to generate two types of
/// annotations:
///   1) Annotations necessary to build interfaces and store them at "~"
///   2) Scattered annotations for how components bind to interfaces
static Optional<DictionaryAttr> parseAugmentedType(
    MLIRContext *context, DictionaryAttr augmentedType, DictionaryAttr root,
    llvm::StringMap<llvm::SmallVector<Attribute>> &newAnnotations,
    StringRef companion, StringAttr name, StringAttr defName,
    Optional<IntegerAttr> id, Optional<StringAttr>(description), Location loc,
    unsigned &annotationID, Twine clazz, Twine path = {}) {

  /// Return a new identifier that can be used to link scattered annotations
  /// together.  This mutates the by-reference parameter annotationID.
  auto newID = [&]() {
    return IntegerAttr::get(IntegerType::get(context, 64), annotationID++);
  };

  /// Optionally unpack a ReferenceTarget encoded as a DictionaryAttr.  Return
  /// either a pair containing the Target string (up to the reference) and an
  /// array of components or none if the input is malformed.  The input
  /// DicionaryAttr encoding is a JSON object of a serialized ReferenceTarget
  /// Scala class.  By example, this is converting:
  ///   ~Foo|Foo>a.b[0]
  /// To:
  ///   {"~Foo|Foo>a", {".b", "[0]"}}
  /// The format of a ReferenceTarget object like:
  ///   circuit: String
  ///   module: String
  ///   path: Seq[(Instance, OfModule)]
  ///   ref: String
  ///   component: Seq[TargetToken]
  auto refTargetToString = [&](DictionaryAttr refTarget)
      -> llvm::Optional<std::pair<std::string, ArrayAttr>> {
    auto circuitAttr =
        tryGetAs<StringAttr>(refTarget, refTarget, "circuit", loc, clazz, path);
    auto moduleAttr =
        tryGetAs<StringAttr>(refTarget, refTarget, "module", loc, clazz, path);
    auto pathAttr =
        tryGetAs<ArrayAttr>(refTarget, refTarget, "path", loc, clazz, path);
    auto componentAttr = tryGetAs<ArrayAttr>(refTarget, refTarget, "component",
                                             loc, clazz, path);
    if (!circuitAttr || !moduleAttr || !pathAttr || !componentAttr)
      return llvm::Optional<std::pair<std::string, ArrayAttr>>();

    // TODO: Enable support for non-local annotations.
    if (!pathAttr.empty()) {
      auto diag = mlir::emitError(
          loc,
          "Annotation '" + clazz + "' with path '" + path +
              "' encodes an unsupported non-local target via the 'path' key.");

      diag.attachNote() << "The encoded target is: " << refTarget;
      return llvm::Optional<std::pair<std::string, ArrayAttr>>();
    }

    auto refAttr =
        tryGetAs<StringAttr>(refTarget, refTarget, "ref", loc, clazz, path);
    SmallVector<Attribute> componentAttrs;
    for (size_t i = 0, e = componentAttr.size(); i != e; ++i) {
      auto cPath = (path + ".component[" + Twine(i) + "]").str();
      auto component = componentAttr[i];
      auto dict = component.dyn_cast_or_null<DictionaryAttr>();
      if (!dict) {
        mlir::emitError(loc,
                        "Annotation '" + clazz + "' with path '" + cPath +
                            " has invalid type (expected DictionaryAttr).");
        return llvm::Optional<std::pair<std::string, ArrayAttr>>();
      }
      auto classAttr =
          tryGetAs<StringAttr>(dict, refTarget, "class", loc, clazz, cPath);
      if (!classAttr)
        return llvm::Optional<std::pair<std::string, ArrayAttr>>();

      auto value = dict.get("value");

      // A subfield like "bar" in "~Foo|Foo>foo.bar".
      if (auto field = value.dyn_cast<StringAttr>()) {
        assert(classAttr.getValue() == "firrtl.annotations.TargetToken$Field" &&
               "A StringAttr target token must be found with a subfield target "
               "token.");
        componentAttrs.push_back(field);
        continue;
      }

      // A subindex like "42" in "~Foo|Foo>foo[42]".
      if (auto index = value.dyn_cast<IntegerAttr>()) {
        assert(classAttr.getValue() == "firrtl.annotations.TargetToken$Index" &&
               "An IntegerAttr target token must be found with a subindex "
               "target token.");
        componentAttrs.push_back(index);
        continue;
      }

      mlir::emitError(loc,
                      "Annotation '" + clazz + "' with path '" + cPath +
                          ".value has unexpected type (should be StringAttr "
                          "for subfield  or IntegerAttr for subindex).")
              .attachNote()
          << "The value received was: " << value << "\n";
      return llvm::Optional<std::pair<std::string, ArrayAttr>>();
    }

    return llvm::Optional<std::pair<std::string, ArrayAttr>>(
        {(Twine("~" + circuitAttr.getValue() + "|" + moduleAttr.getValue() +
                ">" + refAttr.getValue()))
             .str(),
         ArrayAttr::get(context, componentAttrs)});
  };

  auto classAttr =
      tryGetAs<StringAttr>(augmentedType, root, "class", loc, clazz, path);
  if (!classAttr)
    return None;
  StringRef classBase = classAttr.getValue();
  if (!classBase.consume_front("sifive.enterprise.grandcentral.Augmented")) {
    mlir::emitError(loc,
                    "the 'class' was expected to start with "
                    "'sifive.enterprise.grandCentral.Augmented*', but was '" +
                        classAttr.getValue() + "' (Did you misspell it?)")
            .attachNote()
        << "see annotation: " << augmentedType;
    return None;
  }

  // An AugmentedBundleType looks like:
  //   "defName": String
  //   "elements": Seq[AugmentedField]
  if (classBase == "BundleType") {
    defName =
        tryGetAs<StringAttr>(augmentedType, root, "defName", loc, clazz, path);
    if (!defName)
      return None;

    // Each element is an AugmentedField with members:
    //   "name": String
    //   "description": Option[String]
    //   "tpe": AugmenetedType
    SmallVector<Attribute> elements;
    auto elementsAttr =
        tryGetAs<ArrayAttr>(augmentedType, root, "elements", loc, clazz, path);
    if (!elementsAttr)
      return None;
    for (size_t i = 0, e = elementsAttr.size(); i != e; ++i) {
      auto field = elementsAttr[i].dyn_cast_or_null<DictionaryAttr>();
      if (!field) {
        mlir::emitError(
            loc,
            "Annotation '" + Twine(clazz) + "' with path '.elements[" +
                Twine(i) +
                "]' contained an unexpected type (expected a DictionaryAttr).")
                .attachNote()
            << "The received element was: " << elementsAttr[i] << "\n";
        return None;
      }
      auto ePath = (path + ".elements[" + Twine(i) + "]").str();
      auto name = tryGetAs<StringAttr>(field, root, "name", loc, clazz, ePath);
      auto tpe =
          tryGetAs<DictionaryAttr>(field, root, "tpe", loc, clazz, ePath);
      Optional<StringAttr> description = None;
      if (auto maybeDescription = field.get("description"))
        description = maybeDescription.cast<StringAttr>();
      auto eltAttr = parseAugmentedType(
          context, tpe, root, newAnnotations, companion, name, defName, None,
          description, loc, annotationID, clazz, path);
      if (!name || !tpe || !eltAttr)
        return None;

      // Collect information necessary to build a module with this view later.
      // This includes the optional description and name.
      NamedAttrList attrs;
      if (auto maybeDescription = field.get("description"))
        attrs.append("description", maybeDescription.cast<StringAttr>());
      attrs.append("name", name);
      attrs.append("tpe", tpe.getAs<StringAttr>("class"));
      elements.push_back(eltAttr.getValue());
    }
    // Add an annotation that stores information necessary to construct the
    // module for the view.  This needs the name of the module (defName) and the
    // names of the components inside it.
    NamedAttrList attrs;
    attrs.append("class", classAttr);
    attrs.append("defName", defName);
    if (description)
      attrs.append("description", description.getValue());
    attrs.append("elements", ArrayAttr::get(context, elements));
    if (id)
      attrs.append("id", id.getValue());
    attrs.append("name", name);
    return DictionaryAttr::getWithSorted(context, attrs);
  }

  // An AugmentedGroundType looks like:
  //   "ref": ReferenceTarget
  //   "tpe": GroundType
  // The ReferenceTarget is not serialized to a string.  The GroundType will
  // either be an actual FIRRTL ground type or a GrandCentral uninferred type.
  // This can be ignored for us.
  if (classBase == "GroundType") {
    auto maybeTarget =
        refTargetToString(augmentedType.getAs<DictionaryAttr>("ref"));
    if (!maybeTarget) {
      mlir::emitError(loc, "Failed to parse ReferenceTarget").attachNote()
          << "See the full Annotation here: " << root;
      return None;
    }

    auto id = newID();

    auto target = maybeTarget.getValue();
    NamedAttrList elementIface, elementScattered, dontTouch;

    // Populate the annotation for the interface element.
    elementIface.append("class", classAttr);
    if (description)
      elementIface.append("description", description.getValue());
    elementIface.append("id", id);
    elementIface.append("name", name);
    // Populate an annotation that will be scattered onto the element.
    elementScattered.append("class", classAttr);
    elementScattered.append("id", id);
    // Populate a dont touch annotation for the scattered element.
    dontTouch.append(
        "class",
        StringAttr::get(context, "firrtl.transforms.DontTouchAnnotation"));
    // If there are sub-targets, then add these.
    if (target.second) {
      elementScattered.append("target", target.second);
      dontTouch.append("target", target.second);
    }

    newAnnotations[target.first].push_back(
        DictionaryAttr::getWithSorted(context, elementScattered));
    newAnnotations[target.first].push_back(
        DictionaryAttr::getWithSorted(context, dontTouch));

    return DictionaryAttr::getWithSorted(context, elementIface);
  }

  // An AugmentedVectorType looks like:
  //   "elements": Seq[AugmentedType]
  if (classBase == "VectorType") {
    auto elementsAttr =
        tryGetAs<ArrayAttr>(augmentedType, root, "elements", loc, clazz, path);
    if (!elementsAttr)
      return None;
    SmallVector<Attribute> elements;
    for (auto elt : elementsAttr) {
      auto eltAttr = parseAugmentedType(context, elt.cast<DictionaryAttr>(),
                                        root, newAnnotations, companion, name,
                                        StringAttr::get(context, ""), id, None,
                                        loc, annotationID, clazz, path);
      if (!eltAttr)
        return None;
      elements.push_back(eltAttr.getValue());
    }
    NamedAttrList attrs;
    attrs.append("class", classAttr);
    if (description)
      attrs.append("description", description.getValue());
    attrs.append("elements", ArrayAttr::get(context, elements));
    attrs.append("name", name);
    return DictionaryAttr::getWithSorted(context, attrs);
  }

  // Any of the following are known and expected, but are legacy AugmentedTypes
  // do not have a target:
  //   - AugmentedStringType
  //   - AugmentedBooleanType
  //   - AugmentedIntegerType
  //   - AugmentedDoubleType
  bool isIgnorable =
      llvm::StringSwitch<bool>(classBase)
          .Cases("StringType", "BooleanType", "IntegerType", "DoubleType", true)
          .Default(false);
  if (isIgnorable)
    return augmentedType;

  // Anything else is unexpected or a user error if they manually wrote
  // annotations.  Print an error and error out.
  mlir::emitError(loc, "found unknown AugmentedType '" + classAttr.getValue() +
                           "' (Did you misspell it?)")
          .attachNote()
      << "see annotation: " << augmentedType;
  return None;
}

/// Convert known custom FIRRTL Annotations with compound targets to multiple
/// attributes that are attached to IR operations where they have semantic
/// meaning.  This rewrites the input \p annotationMap to convert non-specific
/// Annotations targeting "~" to those targeting something more specific if
/// possible.
bool circt::firrtl::scatterCustomAnnotations(
    llvm::StringMap<ArrayAttr> &annotationMap, CircuitOp circuit,
    unsigned &annotationID, Location loc, size_t &nlaNumber) {
  MLIRContext *context = circuit.getContext();

  // Exit if no anotations exist that target "~". Also ensure a spurious entry
  // is not created in the map.
  if (!annotationMap.count("~"))
    return true;
  // This adds an entry "~" to the map.
  auto nonSpecificAnnotations = annotationMap["~"];

  // Mutable store of new annotations produced.
  llvm::StringMap<llvm::SmallVector<Attribute>> newAnnotations;

  /// Return a new identifier that can be used to link scattered annotations
  /// together.  This mutates the by-reference parameter annotationID.
  auto newID = [&]() {
    return IntegerAttr::get(IntegerType::get(context, 64), annotationID++);
  };

  /// Add a don't touch annotation for a target.
  auto addDontTouch = [&](StringRef target,
                          Optional<ArrayAttr> subfields = {}) {
    NamedAttrList fields;
    fields.append(
        "class",
        StringAttr::get(context, "firrtl.transforms.DontTouchAnnotation"));
    if (subfields)
      fields.append("target", *subfields);
    newAnnotations[target].push_back(
        DictionaryAttr::getWithSorted(context, fields));
  };

  // Loop over all non-specific annotations that target "~".
  //
  //
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
      auto blackBoxAttr =
          tryGetAs<StringAttr>(dict, dict, "blackBox", loc, clazz);
      if (!blackBoxAttr)
        return false;
      auto target = canonicalizeTarget(blackBoxAttr.getValue());
      if (!target)
        return false;
      newAnnotations[target.getValue()].push_back(
          DictionaryAttr::getWithSorted(context, attrs));
      addDontTouch(target.getValue());

      // Process all the taps.
      auto keyAttr = tryGetAs<ArrayAttr>(dict, dict, "keys", loc, clazz);
      if (!keyAttr)
        return false;
      for (size_t i = 0, e = keyAttr.size(); i != e; ++i) {
        auto b = keyAttr[i];
        auto path = ("keys[" + Twine(i) + "]").str();
        auto bDict = b.cast<DictionaryAttr>();
        auto classAttr =
            tryGetAs<StringAttr>(bDict, dict, "class", loc, clazz, path);
        if (!classAttr)
          return false;

        // The "portName" field is common across all sub-types of DataTapKey.
        NamedAttrList port;
        auto portNameAttr =
            tryGetAs<StringAttr>(bDict, dict, "portName", loc, clazz, path);
        if (!portNameAttr)
          return false;
        auto maybePortTarget = canonicalizeTarget(portNameAttr.getValue());
        if (!maybePortTarget)
          return false;
        auto portPair =
            splitAndAppendTarget(port, maybePortTarget.getValue(), context);
        port.append("class", classAttr);
        port.append("id", id);
        addDontTouch(portPair.first, portPair.second);

        if (classAttr.getValue() ==
            "sifive.enterprise.grandcentral.ReferenceDataTapKey") {
          NamedAttrList source;
          auto portID = newID();
          source.append("class", bDict.get("class"));
          source.append("id", id);
          source.append("portID", portID);
          auto sourceAttr =
              tryGetAs<StringAttr>(bDict, dict, "source", loc, clazz, path);
          if (!sourceAttr)
            return false;
          auto maybeSourceTarget = canonicalizeTarget(sourceAttr.getValue());
          if (!maybeSourceTarget)
            return false;
          auto NLATargets = expandNonLocal(*maybeSourceTarget);
          auto leafTarget = splitAndAppendTarget(
              source, std::get<0>(NLATargets.back()), context);
          FlatSymbolRefAttr nlaSym;
          if (NLATargets.size() > 1) {
            nlaSym = buildNLA(circuit, ++nlaNumber, NLATargets);
            source.append("circt.nonlocal", nlaSym);
          }
          source.append("type", StringAttr::get(context, "source"));
          newAnnotations[leafTarget.first].push_back(
              DictionaryAttr::get(context, source));
          addDontTouch(leafTarget.first, leafTarget.second);

          for (int i = 0, e = NLATargets.size() - 1; i < e; ++i) {
            NamedAttrList pathmetadata;
            pathmetadata.append("circt.nonlocal", nlaSym);
            pathmetadata.append("class",
                                StringAttr::get(context, "circt.nonlocal"));
            newAnnotations[std::get<0>(NLATargets[i])].push_back(
                DictionaryAttr::get(context, pathmetadata));
          }

          // Port Annotations generation.
          port.append("portID", portID);
          port.append("type", StringAttr::get(context, "portName"));
          newAnnotations[portPair.first].push_back(
              DictionaryAttr::get(context, port));
          continue;
        }

        if (classAttr.getValue() ==
            "sifive.enterprise.grandcentral.DataTapModuleSignalKey") {
          NamedAttrList module;
          auto portID = newID();
          module.append("class", classAttr);
          module.append("id", id);
          auto internalPathAttr = tryGetAs<StringAttr>(
              bDict, dict, "internalPath", loc, clazz, path);
          auto moduleAttr =
              tryGetAs<StringAttr>(bDict, dict, "module", loc, clazz, path);
          if (!internalPathAttr || !moduleAttr)
            return false;
          module.append("internalPath", internalPathAttr);
          module.append("portID", portID);
          auto moduleTarget = canonicalizeTarget(moduleAttr.getValue());
          if (!moduleTarget)
            return false;
          newAnnotations[moduleTarget.getValue()].push_back(
              DictionaryAttr::getWithSorted(context, module));
          addDontTouch(moduleTarget.getValue());

          // Port Annotations generation.
          port.append("portID", portID);
          newAnnotations[portPair.first].push_back(
              DictionaryAttr::get(context, port));
          continue;
        }

        if (classAttr.getValue() ==
            "sifive.enterprise.grandcentral.DeletedDataTapKey") {
          // Port Annotations generation.
          newAnnotations[portPair.first].push_back(
              DictionaryAttr::get(context, port));
          continue;
        }

        if (classAttr.getValue() ==
            "sifive.enterprise.grandcentral.LiteralDataTapKey") {
          NamedAttrList literal;
          literal.append("class", classAttr);
          auto literalAttr =
              tryGetAs<StringAttr>(bDict, dict, "literal", loc, clazz, path);
          if (!literalAttr)
            return false;
          literal.append("literal", literalAttr);

          // Port Annotaiton generation.
          newAnnotations[portPair.first].push_back(
              DictionaryAttr::get(context, literal));
          continue;
        }

        mlir::emitError(
            loc,
            "Annotation '" + Twine(clazz) + "' with path '" + path + ".class" +
                +"' contained an unknown/unimplemented DataTapKey class '" +
                classAttr.getValue() + "'.")
                .attachNote()
            << "The full Annotation is reprodcued here: " << dict << "\n";
        return false;
      }
      continue;
    }

    if (clazz == "sifive.enterprise.grandcentral.MemTapAnnotation") {
      auto id = newID();
      NamedAttrList attrs;
      auto sourceAttr = tryGetAs<StringAttr>(dict, dict, "source", loc, clazz);
      if (!sourceAttr)
        return false;
      auto target = canonicalizeTarget(sourceAttr.getValue());
      if (!target)
        return false;
      attrs.append(dict.getNamed("class").getValue());
      attrs.append("id", id);
      newAnnotations[target.getValue()].push_back(
          DictionaryAttr::get(context, attrs));
      auto tapsAttr = tryGetAs<ArrayAttr>(dict, dict, "taps", loc, clazz);
      if (!tapsAttr)
        return false;
      for (size_t i = 0, e = tapsAttr.size(); i != e; ++i) {
        auto tap = tapsAttr[i].dyn_cast_or_null<StringAttr>();
        if (!tap) {
          mlir::emitError(
              loc, "Annotation '" + Twine(clazz) + "' with path '.taps[" +
                       Twine(i) +
                       "]' contained an unexpected type (expected a string).")
                  .attachNote()
              << "The full Annotation is reprodcued here: " << dict << "\n";
          return false;
        }
        NamedAttrList foo;
        foo.append("class", dict.get("class"));
        foo.append("id", id);
        foo.append("word", IntegerAttr::get(IntegerType::get(context, 64), i));
        auto canonTarget = canonicalizeTarget(tap.getValue());
        if (!canonTarget)
          return false;
        auto NLATargets = expandNonLocal(*canonTarget);
        auto leafTarget =
            splitAndAppendTarget(foo, std::get<0>(NLATargets.back()), context)
                .first;
        if (NLATargets.size() > 1) {
          buildNLA(circuit, ++nlaNumber, NLATargets);
          foo.append("circt.nonlocal",
                     FlatSymbolRefAttr::get(context, *canonTarget));
        }
        newAnnotations[leafTarget].push_back(DictionaryAttr::get(context, foo));

        for (int i = 0, e = NLATargets.size() - 1; i < e; ++i) {
          NamedAttrList pathmetadata;
          pathmetadata.append("circt.nonlocal",
                              FlatSymbolRefAttr::get(context, *canonTarget));
          pathmetadata.append("class",
                              StringAttr::get(context, "circt.nonlocal"));
          newAnnotations[std::get<0>(NLATargets[i])].push_back(
              DictionaryAttr::get(context, pathmetadata));
        }
      }
      continue;
    }

    if (clazz == "sifive.enterprise.grandcentral.GrandCentralView$"
                 "SerializedViewAnnotation" ||
        clazz == "sifive.enterprise.grandcentral.ViewAnnotation") {
      auto viewAnnotationClass = StringAttr::get(
          context, "sifive.enterprise.grandcentral.ViewAnnotation");
      auto id = newID();
      NamedAttrList companionAttrs, parentAttrs;
      companionAttrs.append("class", viewAnnotationClass);
      companionAttrs.append("id", id);
      companionAttrs.append("type", StringAttr::get(context, "companion"));
      auto viewAttr = tryGetAs<DictionaryAttr>(dict, dict, "view", loc, clazz);
      if (!viewAttr)
        return false;
      auto name = tryGetAs<StringAttr>(dict, dict, "name", loc, clazz);
      if (!name)
        return false;
      companionAttrs.append("name", name);
      auto companionAttr =
          tryGetAs<StringAttr>(dict, dict, "companion", loc, clazz);
      if (!companionAttr)
        return false;
      auto companion = companionAttr.getValue();
      newAnnotations[companion].push_back(
          DictionaryAttr::get(context, companionAttrs));
      auto parentAttr = tryGetAs<StringAttr>(dict, dict, "parent", loc, clazz);
      if (!parentAttr)
        return false;
      parentAttrs.append("class", viewAnnotationClass);
      parentAttrs.append("id", id);
      parentAttrs.append("name", name);
      parentAttrs.append("type", StringAttr::get(context, "parent"));

      newAnnotations[parentAttr.getValue()].push_back(
          DictionaryAttr::get(context, parentAttrs));
      auto prunedAttr = parseAugmentedType(
          context, viewAttr, dict, newAnnotations, companion, name, {}, id, {},
          loc, annotationID, clazz, "view");
      if (!prunedAttr)
        return false;

      newAnnotations["~"].push_back(prunedAttr.getValue());
      continue;
    }

    // Scatter signal driver annotations to the sources *and* the targets of the
    // drives.
    if (clazz == "sifive.enterprise.grandcentral.SignalDriverAnnotation") {
      auto id = newID();

      // Rework the circuit-level annotation to no longer include the
      // information we are scattering away anyway.
      NamedAttrList fields;
      auto annotationsAttr =
          tryGetAs<ArrayAttr>(dict, dict, "annotations", loc, clazz);
      auto circuitAttr =
          tryGetAs<StringAttr>(dict, dict, "circuit", loc, clazz);
      auto circuitPackageAttr =
          tryGetAs<StringAttr>(dict, dict, "circuitPackage", loc, clazz);
      if (!annotationsAttr || !circuitAttr || !circuitPackageAttr)
        return false;
      fields.append("class", classAttr);
      fields.append("id", id);
      fields.append("annotations", annotationsAttr);
      fields.append("circuit", circuitAttr);
      fields.append("circuitPackage", circuitPackageAttr);
      newAnnotations["~"].push_back(DictionaryAttr::get(context, fields));

      // A callback that will scatter every source and sink target pair to the
      // corresponding two ends of the connection.
      llvm::StringSet annotatedModules;
      auto handleTarget = [&](Attribute attr, unsigned i, bool isSource) {
        auto targetId = newID();
        DictionaryAttr targetDict = attr.dyn_cast<DictionaryAttr>();
        if (!targetDict) {
          mlir::emitError(loc, "SignalDriverAnnotation source and sink target "
                               "entries must be dictionaries")
                  .attachNote()
              << "annotation:" << dict << "\n";
          return false;
        }

        // Dig up the two sides of the link.
        auto path = (Twine(clazz) + "." + (isSource ? "source" : "sink") +
                     "Targets[" + Twine(i) + "]")
                        .str();
        auto remoteAttr =
            tryGetAs<StringAttr>(targetDict, dict, "_1", loc, path);
        auto localAttr =
            tryGetAs<StringAttr>(targetDict, dict, "_2", loc, path);
        if (!localAttr || !remoteAttr)
          return false;

        // Build the two annotations.
        for (auto pair : std::array{std::make_pair(localAttr, true),
                                    std::make_pair(remoteAttr, false)}) {
          auto canonTarget = canonicalizeTarget(pair.first.getValue());
          if (!canonTarget)
            return false;

          // HACK: Ignore the side of the connection that targets the *other*
          // circuit. We do this by checking whether the canonicalized target
          // begins with `~CircuitName|`. If it doesn't, we skip.
          // TODO: Once we properly support multiple circuits, this can go and
          // the annotation can scatter properly.
          StringRef prefix(*canonTarget);
          if (!(prefix.consume_front("~") &&
                prefix.consume_front(circuit.name()) &&
                prefix.consume_front("|"))) {
            continue;
          }

          // Assemble the annotation on this side of the connection.
          NamedAttrList fields;
          fields.append("class", classAttr);
          fields.append("id", id);
          fields.append("targetId", targetId);
          fields.append("peer", pair.second ? remoteAttr : localAttr);
          fields.append("side", StringAttr::get(
                                    context, pair.second ? "local" : "remote"));
          fields.append("dir",
                        StringAttr::get(context, isSource ? "source" : "sink"));

          // Handle subfield and non-local targets.
          auto NLATargets = expandNonLocal(*canonTarget);
          auto leafTarget = splitAndAppendTarget(
              fields, std::get<0>(NLATargets.back()), context);
          if (NLATargets.size() > 1) {
            buildNLA(circuit, ++nlaNumber, NLATargets);
            fields.append("circt.nonlocal",
                          FlatSymbolRefAttr::get(context, *canonTarget));
          }
          newAnnotations[leafTarget.first].push_back(
              DictionaryAttr::get(context, fields));

          // Add a don't touch annotation to whatever this annotation targets.
          addDontTouch(leafTarget.first, leafTarget.second);

          // Keep track of the enclosing module.
          annotatedModules.insert(
              (StringRef(std::get<0>(NLATargets.back())).split("|").first +
               "|" + std::get<1>(NLATargets.back()))
                  .str());

          // Annotate instances along the NLA path.
          for (int i = 0, e = NLATargets.size() - 1; i < e; ++i) {
            NamedAttrList fields;
            fields.append("circt.nonlocal",
                          FlatSymbolRefAttr::get(context, *canonTarget));
            fields.append("class", StringAttr::get(context, "circt.nonlocal"));
            newAnnotations[std::get<0>(NLATargets[i])].push_back(
                DictionaryAttr::get(context, fields));
          }
        }

        return true;
      };

      // Handle the source and sink targets.
      auto sourcesAttr =
          tryGetAs<ArrayAttr>(dict, dict, "sourceTargets", loc, clazz);
      auto sinksAttr =
          tryGetAs<ArrayAttr>(dict, dict, "sinkTargets", loc, clazz);
      if (!sourcesAttr || !sinksAttr)
        return false;
      unsigned i = 0;
      for (auto attr : sourcesAttr)
        if (!handleTarget(attr, i++, true))
          return false;
      i = 0;
      for (auto attr : sinksAttr)
        if (!handleTarget(attr, i++, false))
          return false;

      // Indicate which modules have embedded `SignalDriverAnnotation`s.
      for (auto &module : annotatedModules) {
        NamedAttrList fields;
        fields.append("class", classAttr);
        fields.append("id", id);
        newAnnotations[module.getKey()].push_back(
            DictionaryAttr::get(context, fields));
      }

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

    attrs.push_back(DictionaryAttr::get(context, metadata));
  }

  return true;
}
