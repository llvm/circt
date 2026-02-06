//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generates FIRRTL annotations helpers and documentation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/IndentedOstream.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <variant>

using namespace llvm;
using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Parameter
//===----------------------------------------------------------------------===//

struct Parameter;

/// Simple parameter types (just type tags).
struct StringType {
  static StringRef getJSONTypeName() { return "string"; }
};
struct IntegerType {
  static StringRef getJSONTypeName() { return "integer"; }
};
struct BooleanType {
  static StringRef getJSONTypeName() { return "boolean"; }
};
struct ArrayType {
  /// Name of the element type (e.g., "StringParam").
  std::string elementTypeName;

  static StringRef getJSONTypeName() { return "array"; }
  StringRef getElementTypeName() const { return elementTypeName; }
};
struct TargetType {
  static StringRef getJSONTypeName() { return "target"; }
};

/// Enum type with allowed values.
struct EnumType {
  SmallVector<StringRef> allowedValues;

  static StringRef getJSONTypeName() { return "string"; }
  ArrayRef<StringRef> getAllowedValues() const { return allowedValues; }
};

/// Union type with allowed object types.
struct UnionType {
  SmallVector<StringRef> allowedTypes;

  static StringRef getJSONTypeName() { return "object"; }
  ArrayRef<StringRef> getAllowedTypes() const { return allowedTypes; }
};

/// Object type with description and fields.
struct ObjectType {
  std::string description;
  std::vector<Parameter> fields;

  static StringRef getJSONTypeName() { return "object"; }
  StringRef getDescription() const { return description; }
  ArrayRef<Parameter> getFields() const { return fields; }
};

/// Annotation type - references another annotation definition as a member type.
/// This is distinct from ObjectType in that it refers to a named annotation
/// definition rather than an inline object structure.
struct AnnotationType {
  /// TableGen def name (e.g., "AugmentedGroundType").
  std::string annotationName;

  /// JSON class name.
  std::string className;

  static StringRef getJSONTypeName() { return "object"; }
  StringRef getAnnotationName() const { return annotationName; }
  StringRef getClassName() const { return className; }
};

/// Variant type for all possible parameter types. The std::monostate
/// alternative represents an uninitialized state.
using ParameterTypeVariant =
    std::variant<std::monostate, StringType, IntegerType, BooleanType,
                 ArrayType, TargetType, EnumType, UnionType, ObjectType,
                 AnnotationType>;

/// Represents a parameter with its name, type, and metadata.
struct Parameter {
  StringRef name;
  std::string description;
  bool required = true;
  std::string defaultValue;

  /// Type-safe variant holding the actual parameter type.
  ParameterTypeVariant type;

  Parameter() = default;

  /// Get the JSON type name from the variant.
  StringRef getJSONTypeName() const {
    return std::visit(
        [](auto &&arg) -> StringRef {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, std::monostate>)
            return "unknown";
          else
            return arg.getJSONTypeName();
        },
        type);
  }
};

//===----------------------------------------------------------------------===//
// TargetTypeDef
//===----------------------------------------------------------------------===//

/// Represents a FIRRTL target type. This is a pure AST node with no TableGen
/// dependencies.
struct TargetTypeDef {
  /// The target type name (e.g., "Module", "Wire").
  StringRef name;

  /// The MLIR operation name (e.g., "FModuleOp", "WireOp").
  StringRef operation;

  TargetTypeDef() = default;
  TargetTypeDef(StringRef name, StringRef operation)
      : name(name), operation(operation) {}
};

//===----------------------------------------------------------------------===//
// AnnotationDef
//===----------------------------------------------------------------------===//

/// Represents a FIRRTL annotation. This is a pure AST node with no TableGen
/// dependencies. It captures all the information needed to generate code and
/// documentation for an annotation.
struct AnnotationDef {
  StringRef className;
  std::string description;
  SmallVector<Parameter> parameters;

  /// The annotation type determines how the annotation is resolved and applied.
  enum class AnnotationType { NoTarget, SingleTarget, Ignored } annoType;

  /// These fields are only used for SingleTargetAnnotation.
  bool allowNonLocal = false;
  bool allowPortTargets = false;

  /// MLIR operation names that this annotation can target.
  SmallVector<StringRef> targets;

  /// Custom handler function name, if specified.
  std::string customHandler;

  /// Source location for error reporting.
  ArrayRef<SMLoc> loc;

  AnnotationDef() = default;

  /// Convert className to a valid C++ identifier for variable names.
  /// This creates names like "fooAnnoClass" from "com.example.FooAnnotation".
  std::string makeIdentifier() const;

  /// Get the first line of the description for brief comments.
  StringRef getFirstLineOfDescription() const;

  /// Get the resolver string for this annotation type.
  StringRef getResolver() const;

  /// Get the applier string for this annotation type.
  std::string getApplier() const;
};

std::string AnnotationDef::makeIdentifier() const {
  std::string result;

  // Remove trailing '$' if present
  auto cleanName = className;
  if (cleanName.ends_with("$"))
    cleanName = cleanName.drop_back();

  // Take the last component after the last '.' or '$'
  auto lastDot = cleanName.rfind('.');
  auto lastDollar = cleanName.rfind('$');
  size_t start = 0;

  // Find the position of the last separator
  if (lastDot != StringRef::npos && lastDollar != StringRef::npos)
    start = std::max(lastDot, lastDollar) + 1;
  else if (lastDot != StringRef::npos)
    start = lastDot + 1;
  else if (lastDollar != StringRef::npos)
    start = lastDollar + 1;

  auto baseName = cleanName.substr(start);
  result = baseName.str();

  // Ensure it ends with "AnnoClass"
  StringRef resultRef(result);
  if (resultRef.ends_with("Annotation"))
    result = result.substr(0, result.size() - 10) + "AnnoClass";
  else if (resultRef.ends_with("Anno"))
    result = result.substr(0, result.size() - 4) + "AnnoClass";
  else
    result += "AnnoClass";

  // Make first letter lowercase
  if (!result.empty())
    result[0] = std::tolower(result[0]);

  return result;
}

StringRef AnnotationDef::getFirstLineOfDescription() const {
  if (description.empty())
    return "";

  auto desc = StringRef(description).trim();
  auto firstNewline = desc.find('\n');
  return firstNewline == StringRef::npos ? desc : desc.substr(0, firstNewline);
}

StringRef AnnotationDef::getResolver() const {
  if (annoType == AnnotationType::NoTarget ||
      annoType == AnnotationType::Ignored)
    return "noResolve";
  return "stdResolve";
}

std::string AnnotationDef::getApplier() const {
  // If customHandler is specified, use it directly (manual override)
  if (!customHandler.empty())
    return customHandler;

  if (annoType == AnnotationType::Ignored) {
    // IgnoredAnnotation - always drop
    return "drop";
  }
  if (annoType == AnnotationType::NoTarget) {
    // NoTargetAnnotation - apply to CircuitOp
    return "applyWithoutTarget<false, CircuitOp>";
  }
  if (annoType == AnnotationType::SingleTarget) {
    // SingleTargetAnnotation - auto-generate from metadata
    // Build the handler based on targets, allowNonLocal, allowPortTargets

    if (!targets.empty()) {
      // Specific target types - use the operation names directly
      std::string opTypes;
      for (size_t i = 0; i < targets.size(); ++i) {
        if (i > 0)
          opTypes += ", ";
        opTypes += targets[i].str();
      }

      // Build template arguments: <allowNonLocal, allowPortTargets,
      // OpTypes...>
      auto applier = std::string("applyWithoutTarget<");
      applier += allowNonLocal ? "true" : "false";
      applier += ", ";
      applier += allowPortTargets ? "true" : "false";
      applier += ", ";
      applier += opTypes;
      applier += ">";
      return applier;
    }
    // No specific target types - any named target is allowed
    auto applier = std::string("applyWithoutTarget<");
    applier += allowNonLocal ? "true" : "false";
    applier += ">";
    return applier;
  }
  // Fallback - should not happen
  PrintWarning(loc, "Unknown annotation type for '" + className + "'");
  return "drop";
}

//===----------------------------------------------------------------------===//
// TableGen Parser
//===----------------------------------------------------------------------===//

/// Check if a record is or inherits from a given class.
static bool isOrInheritsFrom(const Record *rec, StringRef className) {
  return rec->getName() == className || rec->isSubClassOf(className);
}

/// Unwrap Optional<> and Doc<> wrappers to get the base type.
/// Also extracts documentation if present.
static const Record *unwrapParamType(const Record *paramTypeRec,
                                     std::string &documentation) {
  // First check for Doc<> wrapper and extract documentation
  if (isOrInheritsFrom(paramTypeRec, "Doc")) {
    documentation =
        paramTypeRec->getValueAsOptionalString("documentation").value_or("");
    if (auto *baseTypeInit = paramTypeRec->getValue("baseType")) {
      if (auto *defInit = dyn_cast<DefInit>(baseTypeInit->getValue())) {
        paramTypeRec = defInit->getDef();
      }
    }
  }

  // Then unwrap Optional<> if present
  if (isOrInheritsFrom(paramTypeRec, "Optional")) {
    if (auto *baseTypeInit = paramTypeRec->getValue("baseType")) {
      if (auto *defInit = dyn_cast<DefInit>(baseTypeInit->getValue())) {
        paramTypeRec = defInit->getDef();
      }
    }
  }

  return paramTypeRec;
}

/// Parse a target type definition from a TableGen record into the AST.
static TargetTypeDef parseTargetTypeFromRecord(const Record *def) {
  TargetTypeDef targetType;
  targetType.name = def->getName();
  targetType.operation = def->getValueAsString("operation");
  return targetType;
}

/// Parse a parameter from a TableGen record into the AST. This is used
/// recursively for nested object fields.
// NOLINTNEXTLINE(misc-no-recursion)
static void parseParameterFromRecord(const RecordKeeper &records,
                                     const Record *paramTypeRec,
                                     StringRef paramName, Parameter &param) {
  param.name = paramName;

  // Unwrap Doc<> and Optional<> wrappers, extracting documentation
  std::string documentation;
  auto *unwrapped = unwrapParamType(paramTypeRec, documentation);
  param.description = documentation;

  // Check if this is wrapped in Optional<> (need to check before unwrapping)
  auto *afterDoc = paramTypeRec;
  if (isOrInheritsFrom(paramTypeRec, "Doc")) {
    if (auto *baseTypeInit = paramTypeRec->getValue("baseType")) {
      if (auto *defInit = dyn_cast<DefInit>(baseTypeInit->getValue())) {
        afterDoc = defInit->getDef();
      }
    }
  }
  param.required = !isOrInheritsFrom(afterDoc, "Optional");
  param.defaultValue =
      afterDoc->getValueAsOptionalString("defaultValue").value_or("");

  // Parse the type based on the unwrapped record
  if (isOrInheritsFrom(unwrapped, "UnionParam")) {
    // Parse UnionType
    UnionType unionType;
    const auto *allowedTypesInit =
        unwrapped->getValueAsListInit("allowedTypes");
    for (auto *val : allowedTypesInit->getElements()) {
      if (auto *defInit = dyn_cast<DefInit>(val))
        unionType.allowedTypes.push_back(defInit->getDef()->getName());
    }
    param.type = unionType;

  } else if (isOrInheritsFrom(unwrapped, "EnumParam")) {
    // Parse EnumType
    EnumType enumType;
    const auto *allowedValuesInit =
        unwrapped->getValueAsListInit("allowedValues");
    for (auto *val : allowedValuesInit->getElements()) {
      if (auto *str = dyn_cast<StringInit>(val))
        enumType.allowedValues.push_back(str->getValue());
    }
    param.type = enumType;

  } else if (isOrInheritsFrom(unwrapped, "ObjectParam")) {
    // Parse ObjectType
    ObjectType objType;
    objType.description =
        unwrapped->getValueAsOptionalString("description").value_or("");

    // Parse nested fields from objectFields DAG
    if (auto *objectFieldsInit = unwrapped->getValue("objectFields")) {
      if (auto *dagInit = dyn_cast<DagInit>(objectFieldsInit->getValue())) {
        for (unsigned i = 0; i < dagInit->getNumArgs(); ++i) {
          auto fieldName = dagInit->getArgNameStr(i);
          if (fieldName.empty())
            continue;

          auto *fieldInit = dagInit->getArg(i);
          const Record *fieldTypeRec = nullptr;

          if (auto *defInit = dyn_cast<DefInit>(fieldInit))
            fieldTypeRec = defInit->getDef();

          if (!fieldTypeRec)
            continue;

          // Recursively parse the nested field
          Parameter nestedParam;
          parseParameterFromRecord(records, fieldTypeRec, fieldName,
                                   nestedParam);
          objType.fields.push_back(nestedParam);
        }
      }
    }
    param.type = objType;

  } else if (isOrInheritsFrom(unwrapped, "StringParam")) {
    param.type = StringType{};
  } else if (isOrInheritsFrom(unwrapped, "IntegerParam")) {
    param.type = IntegerType{};
  } else if (isOrInheritsFrom(unwrapped, "BooleanParam")) {
    param.type = BooleanType{};
  } else if (isOrInheritsFrom(unwrapped, "ArrayParam")) {
    // Parse ArrayType with element type
    ArrayType arrayType;
    if (auto *elementInit = unwrapped->getValue("element")) {
      if (auto *defInit = dyn_cast<DefInit>(elementInit->getValue())) {
        arrayType.elementTypeName = defInit->getDef()->getName().str();
      }
    }
    param.type = arrayType;
  } else if (unwrapped->getName() == "TargetParam") {
    param.type = TargetType{};
  } else if (isOrInheritsFrom(unwrapped, "Annotation")) {
    // Parse AnnotationType - an annotation used as a member type
    AnnotationType annoType;
    annoType.annotationName = unwrapped->getName().str();
    annoType.className =
        unwrapped->getValueAsOptionalString("className").value_or("");
    param.type = annoType;
  } else {
    // Unknown type - leave as monostate
    param.type = std::monostate{};
  }
}

/// Parse an annotation definition from a TableGen record into the AST.
static AnnotationDef parseAnnotationFromRecord(const RecordKeeper &records,
                                               const Record *def) {
  AnnotationDef anno;
  anno.loc = def->getLoc();
  anno.className = def->getValueAsString("className");
  anno.description = def->getValueAsOptionalString("description").value_or("");
  anno.customHandler = def->getValueAsString("customHandler").str();

  // Detect annotation type from base class
  if (def->isSubClassOf("IgnoredAnnotation")) {
    anno.annoType = AnnotationDef::AnnotationType::Ignored;
  } else if (def->isSubClassOf("NoTargetAnnotation")) {
    anno.annoType = AnnotationDef::AnnotationType::NoTarget;
  } else if (def->isSubClassOf("SingleTargetAnnotation")) {
    anno.annoType = AnnotationDef::AnnotationType::SingleTarget;
    anno.allowNonLocal = def->getValueAsBit("allowNonLocal");
    anno.allowPortTargets = def->getValueAsBit("allowPortTargets");

    // Get target types and their operation names
    const auto *targetsInit = def->getValueAsListInit("targets");
    for (auto *val : targetsInit->getElements()) {
      if (auto *defInit = dyn_cast<DefInit>(val)) {
        // Parse the TargetType record
        const Record *targetTypeRec = defInit->getDef();
        TargetTypeDef targetType = parseTargetTypeFromRecord(targetTypeRec);
        if (!targetType.operation.empty()) {
          anno.targets.push_back(targetType.operation);
        }
      }
    }
  } else {
    // Unknown annotation type - default to NoTarget
    anno.annoType = AnnotationDef::AnnotationType::NoTarget;
  }

  // For SingleTargetAnnotation, automatically add 'target' parameter first.
  // This is a synthetic parameter that is not defined in TableGen, but is
  // always present in the JSON representation.
  if (anno.annoType == AnnotationDef::AnnotationType::SingleTarget) {
    Parameter targetParam;
    targetParam.name = "target";
    targetParam.type = TargetType{};
    targetParam.description = "Target of the annotation";
    targetParam.required = true;
    anno.parameters.push_back(targetParam);
  }

  // Get parameters from the members dag
  auto *membersDag = def->getValueAsDag("members");
  if (membersDag && membersDag->getNumArgs() > 0) {
    for (unsigned i = 0; i < membersDag->getNumArgs(); ++i) {
      StringRef paramName = membersDag->getArgNameStr(i);
      if (paramName.empty())
        continue;

      // Get the parameter type (could be ParamType or Doc<ParamType>)
      const Init *argInit = membersDag->getArg(i);
      const Record *paramTypeRec = nullptr;

      if (auto *defInit = dyn_cast<DefInit>(argInit))
        paramTypeRec = defInit->getDef();

      if (!paramTypeRec)
        continue;

      // Extract parameter information from the type record
      Parameter param;
      parseParameterFromRecord(records, paramTypeRec, paramName, param);
      anno.parameters.push_back(param);
    }
  }

  return anno;
}

} // anonymous namespace

static bool emitMarkdownDocs(const RecordKeeper &records, raw_ostream &os) {
  os << "<!-- Autogenerated by circt-tblgen; don't manually edit -->\n\n";
  os << "# FIRRTL Annotations\n\n";
  os << "This document describes the annotations supported by CIRCT's FIRRTL "
        "compiler.\n\n";

  // Collect all top-level ObjectParam definitions
  SmallVector<const Record *> objectParams;
  for (const auto *def : records.getAllDerivedDefinitions("ObjectParam")) {
    // Only include top-level definitions (those with names, not anonymous
    // instances)
    if (!def->getName().starts_with("anonymous_"))
      objectParams.push_back(def);
  }

  // Sort by name
  sort(objectParams, [](const Record *a, const Record *b) {
    return a->getName() < b->getName();
  });

  // Emit ObjectParam type definitions if any exist
  if (!objectParams.empty()) {
    os << "## Object Type Definitions\n\n";
    os << "These are reusable object types that can be used in annotation "
          "parameters.\n\n";

    for (const auto *def : objectParams) {
      os << "### " << def->getName() << "\n\n";

      // Parse the ObjectType from the record
      ObjectType objType;
      objType.description =
          def->getValueAsOptionalString("description").value_or("");

      // Parse fields from objectFields DAG
      if (auto *objectFieldsInit = def->getValue("objectFields")) {
        if (auto *dagInit = dyn_cast<DagInit>(objectFieldsInit->getValue())) {
          for (unsigned i = 0; i < dagInit->getNumArgs(); ++i) {
            auto fieldName = dagInit->getArgNameStr(i);
            if (fieldName.empty())
              continue;

            auto *fieldInit = dagInit->getArg(i);
            const Record *fieldTypeRec = nullptr;

            if (auto *defInit = dyn_cast<DefInit>(fieldInit))
              fieldTypeRec = defInit->getDef();

            if (!fieldTypeRec)
              continue;

            Parameter param;
            parseParameterFromRecord(records, fieldTypeRec, fieldName, param);
            objType.fields.push_back(param);
          }
        }
      }

      // Print description
      if (!objType.description.empty()) {
        raw_indented_ostream(os).printReindented(
            StringRef(objType.description).rtrim());
        os << "\n\n";
      }

      // Emit parameter table
      os << "| Property | Type | Description |\n";
      os << "| -------- | ---- | ----------- |\n";

      // Helper lambda to print a parameter row (used recursively for nested
      // object fields).
      std::function<void(const Parameter &, StringRef)> printParameter;
      printParameter = [&](const Parameter &param, StringRef prefix) {
        os << "| " << prefix << param.name << " | " << param.getJSONTypeName()
           << " | ";

        if (!param.description.empty())
          os << param.description;

        // Add allowed values if present (for EnumParam)
        if (auto *enumType = std::get_if<EnumType>(&param.type)) {
          auto allowedValues = enumType->getAllowedValues();
          if (!allowedValues.empty()) {
            os << " (";
            bool first = true;
            for (auto val : allowedValues) {
              if (!first)
                os << ", ";
              os << "`" << val << "`";
              first = false;
            }
            os << ")";
          }
        }

        // Add allowed types if present (for UnionParam)
        if (auto *unionType = std::get_if<UnionType>(&param.type)) {
          auto allowedTypes = unionType->getAllowedTypes();
          if (!allowedTypes.empty()) {
            os << " (one of: ";
            bool first = true;
            for (auto typeName : allowedTypes) {
              if (!first)
                os << ", ";
              os << "`" << typeName << "`";
              first = false;
            }
            os << ")";
          }
        }

        // Mark as optional if not required
        if (!param.required) {
          os << " (optional";
          if (!param.defaultValue.empty())
            os << ", default: `" << param.defaultValue << "`";
          os << ")";
        }

        os << " |\n";

        // If this is an object with structured fields, print them as nested
        // rows
        if (auto *objType = std::get_if<ObjectType>(&param.type)) {
          auto objFields = objType->getFields();
          if (!objFields.empty()) {
            auto nestedPrefix = (prefix + param.name + ".").str();
            for (const auto &field : objFields) {
              printParameter(field, nestedPrefix);
            }
          }
        }
      };

      // Print all fields
      for (const auto &field : objType.fields) {
        printParameter(field, "");
      }

      os << "\n";
    }
  }

  // Collect all annotations (from all base classes)
  SmallVector<AnnotationDef, 0> annotations;
  for (const auto *def : records.getAllDerivedDefinitions("Annotation"))
    annotations.push_back(parseAnnotationFromRecord(records, def));

  // Sort by className
  sort(annotations, [](const AnnotationDef &a, const AnnotationDef &b) {
    return a.className < b.className;
  });

  // Emit all annotations
  os << "## FIRRTL Annotations\n\n";

  for (const auto &anno : annotations) {
    // Emit header
    os << "### " << anno.className << "\n\n";

    // Emit parameter table
    os << "| Property | Type | Description |\n";
    os << "| -------- | ---- | ----------- |\n";
    os << "| class | string | `" << anno.className << "` |\n";

    // Add target if annotation has one
    bool hasTargetField =
        (anno.annoType == AnnotationDef::AnnotationType::SingleTarget);

    if (hasTargetField) {
      os << "| target | string | ";

      // For SingleTargetAnnotation, show allowed target types
      if (!anno.targets.empty()) {
        bool first = true;
        for (auto opName : anno.targets) {
          if (!first)
            os << ", ";
          // Remove "Op" suffix for display (e.g., "WireOp" -> "Wire")
          StringRef displayName = opName;
          if (displayName.ends_with("Op"))
            displayName = displayName.drop_back(2);
          os << displayName;
          first = false;
        }
        os << " target";
      } else {
        os << "Any named target";
      }
      os << " |\n";
    }

    // Helper lambda to print a parameter row (used recursively for object
    // fields)
    std::function<void(const Parameter &, StringRef)> printParameter;
    printParameter = [&](const Parameter &param, StringRef prefix) {
      os << "| " << prefix << param.name << " | " << param.getJSONTypeName()
         << " | ";

      if (!param.description.empty())
        os << param.description;

      // Add allowed values if present (for EnumParam)
      if (auto *enumType = std::get_if<EnumType>(&param.type)) {
        auto allowedValues = enumType->getAllowedValues();
        if (!allowedValues.empty()) {
          os << " (";
          bool first = true;
          for (auto val : allowedValues) {
            if (!first)
              os << ", ";
            os << "`" << val << "`";
            first = false;
          }
          os << ")";
        }
      }

      // Add allowed types if present (for UnionParam)
      if (auto *unionType = std::get_if<UnionType>(&param.type)) {
        auto allowedTypes = unionType->getAllowedTypes();
        if (!allowedTypes.empty()) {
          os << " (one of: ";
          bool first = true;
          for (auto typeName : allowedTypes) {
            if (!first)
              os << ", ";
            os << "`" << typeName << "`";
            first = false;
          }
          os << ")";
        }
      }

      // Add annotation reference if present (for Annotation as member type)
      if (auto *annoType = std::get_if<AnnotationType>(&param.type)) {
        os << " (annotation: `" << annoType->getAnnotationName() << "`)";
      }

      // Add element type if present (for ArrayParam)
      if (auto *arrayType = std::get_if<ArrayType>(&param.type)) {
        auto elementTypeName = arrayType->getElementTypeName();
        if (!elementTypeName.empty()) {
          // Show a cleaner name for anonymous ObjectParam records
          if (elementTypeName.starts_with("anonymous_"))
            os << " (elements: `object`)";
          else
            os << " (elements: `" << elementTypeName << "`)";
        }
      }

      // Mark as optional if not required
      if (!param.required) {
        os << " (optional";
        if (!param.defaultValue.empty())
          os << ", default: `" << param.defaultValue << "`";
        os << ")";
      }

      os << " |\n";

      // If this is an object with structured fields, print them as nested rows
      if (auto *objType = std::get_if<ObjectType>(&param.type)) {
        auto objFields = objType->getFields();
        if (!objFields.empty()) {
          std::string nestedPrefix = (prefix + param.name + ".").str();
          for (const auto &field : objFields) {
            printParameter(field, nestedPrefix);
          }
        }
      }
    };

    // Add parameters
    for (const auto &param : anno.parameters) {
      // Skip "target" parameter if we already added it above
      if (hasTargetField && param.name == "target")
        continue;

      printParameter(param, "");
    }

    os << "\n";

    // Emit description
    if (!anno.description.empty()) {
      raw_indented_ostream(os).printReindented(
          StringRef(anno.description).rtrim());
      os << "\n\n";
    }
  }

  return false;
}

static GenRegistration genDocs("firrtl-annotations-doc",
                               "Generate documentation for FIRRTL annotations",
                               emitMarkdownDocs);

//===----------------------------------------------------------------------===//
// Generate AnnotationDetails.h
//===----------------------------------------------------------------------===//

static bool emitAnnotationDetails(const RecordKeeper &records,
                                  raw_ostream &os) {
  // Get all annotation definitions
  SmallVector<AnnotationDef, 0> annotations;
  for (const auto *def : records.getAllDerivedDefinitions("Annotation")) {
    annotations.push_back(parseAnnotationFromRecord(records, def));
  }

  // Sort by className for consistent output
  llvm::sort(annotations, [](const AnnotationDef &a, const AnnotationDef &b) {
    return a.className < b.className;
  });

  // Emit header comment
  os << "//===- FIRRTLAnnotationDetails.h.inc - Annotation Details "
        "---------------===//\n";
  os << "//\n";
  os << "// Part of the LLVM Project, under the Apache License v2.0 with LLVM "
        "Exceptions.\n";
  os << "// See https://llvm.org/LICENSE.txt for license information.\n";
  os << "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n";
  os << "//\n";
  os << "//"
        "===-------------------------------------------------------------------"
        "---"
        "===//\n";
  os << "//\n";
  os << "// Auto-generated annotation class name constants.\n";
  os << "//\n";
  os << "// THIS FILE IS AUTOMATICALLY GENERATED. DO NOT EDIT!\n";
  os << "// Generated from FIRRTLAnnotations.td using circt-tblgen.\n";
  os << "//\n";
  os << "//"
        "===-------------------------------------------------------------------"
        "---"
        "===//\n\n";

  // Note: No header guards or namespace - this is an .inc file meant to be
  // included inside the circt::firrtl namespace

  // Emit all annotations
  for (const auto &anno : annotations) {
    auto varName = anno.makeIdentifier();

    // Add comment if there's a description
    auto firstLine = anno.getFirstLineOfDescription();
    if (!firstLine.empty())
      os << "/// " << firstLine.trim() << "\n";

    os << "constexpr const char *" << varName << " = \"" << anno.className
       << "\";\n";
  }

  os << "\n";

  // Note: No closing namespace or header guard - this is an .inc file
  return false;
}

static GenRegistration genDetails("firrtl-annotations-details",
                                  "Generate AnnotationDetails.h header",
                                  emitAnnotationDetails);

//===----------------------------------------------------------------------===//
// Annotation Records Generator (for LowerAnnotations.cpp)
//===----------------------------------------------------------------------===//

static bool emitAnnotationRecords(const RecordKeeper &records,
                                  raw_ostream &os) {
  // Collect all annotations (from all base classes)
  SmallVector<AnnotationDef, 0> annotations;
  for (const auto *def : records.getAllDerivedDefinitions("Annotation"))
    annotations.push_back(parseAnnotationFromRecord(records, def));

  // Emit header comment
  os << "//===- FIRRTLAnnotationRecords.h.inc - Annotation Records "
        "-----------------===//\n";
  os << "//\n";
  os << "// Part of the LLVM Project, under the Apache License v2.0 with LLVM "
        "Exceptions.\n";
  os << "// See https://llvm.org/LICENSE.txt for license information.\n";
  os << "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n";
  os << "//\n";
  os << "//"
        "===------------------------------------------------------------------"
        "----===//\n";
  os << "//\n";
  os << "// Auto-generated annotation handler records for LowerAnnotations "
        "pass.\n";
  os << "//\n";
  os << "// THIS FILE IS AUTOMATICALLY GENERATED. DO NOT EDIT!\n";
  os << "// Generated from FIRRTLAnnotations.td using circt-tblgen.\n";
  os << "//\n";
  os << "//"
        "===------------------------------------------------------------------"
        "----===//\n\n";

  // Emit the macro-guarded annotation record list
  os << "#ifdef GET_ANNOTATION_RECORD_LIST\n\n";

  // Emit annotation records
  for (const auto &anno : annotations) {
    auto varName = anno.makeIdentifier();
    auto resolver = anno.getResolver();
    auto applier = anno.getApplier();

    os << "  {" << varName << ", {" << resolver << ", " << applier << "}},\n";
  }

  os << "\n#undef GET_ANNOTATION_RECORD_LIST\n";
  os << "#endif // GET_ANNOTATION_RECORD_LIST\n";

  return false;
}

static GenRegistration genRecords("firrtl-annotation-records",
                                  "Generate annotation handler records",
                                  emitAnnotationRecords);
