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
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace mlir;

namespace {

// Forward declarations
struct ParameterDef;
static void parseParameterFromRecord(const Record *paramTypeRec,
                                     StringRef paramName, ParameterDef &param);

// Represents an annotation parameter from TableGen
struct ParameterDef {
  StringRef name;
  StringRef type;
  StringRef description;
  bool required = true;
  StringRef defaultValue;
  std::vector<StringRef> allowedValues;

  // For structured ObjectParam: nested fields
  std::vector<ParameterDef> objectFields;

  // Default constructor
  ParameterDef() = default;
};

// Helper function to parse a parameter from a TableGen record
// This is used recursively for nested object fields
// NOLINTNEXTLINE(misc-no-recursion)
static void parseParameterFromRecord(const Record *paramTypeRec,
                                     StringRef paramName, ParameterDef &param) {
  param.name = paramName;
  param.type = paramTypeRec->getValueAsString("type");
  param.description =
      paramTypeRec->getValueAsOptionalString("documentation").value_or("");

  // Check if this is an Optional<> wrapper (has isRequired field set to 0)
  param.required = true;
  if (auto *isReqInit = paramTypeRec->getValue("isRequired")) {
    if (auto *bitInit = dyn_cast<BitInit>(isReqInit->getValue()))
      param.required = bitInit->getValue();
  }
  param.defaultValue =
      paramTypeRec->getValueAsOptionalString("defaultValue").value_or("");

  // Get allowed values if present
  const auto *allowedValuesInit =
      paramTypeRec->getValueAsListInit("allowedValues");
  for (auto *val : allowedValuesInit->getElements()) {
    if (auto *str = dyn_cast<StringInit>(val))
      param.allowedValues.push_back(str->getValue());
  }

  // For ObjectParam, parse nested fields from objectFields DAG
  if (param.type == "object") {
    if (auto *objectFieldsInit = paramTypeRec->getValue("objectFields")) {
      if (auto *dagInit = dyn_cast<DagInit>(objectFieldsInit->getValue())) {
        // Parse each field in the DAG
        for (unsigned i = 0; i < dagInit->getNumArgs(); ++i) {
          StringRef fieldName = dagInit->getArgNameStr(i);
          if (fieldName.empty())
            continue;

          const Init *fieldInit = dagInit->getArg(i);
          const Record *fieldTypeRec = nullptr;

          if (auto *defInit = dyn_cast<DefInit>(fieldInit))
            fieldTypeRec = defInit->getDef();

          if (!fieldTypeRec)
            continue;

          // Recursively parse the nested field
          ParameterDef nestedParam;
          parseParameterFromRecord(fieldTypeRec, fieldName, nestedParam);
          param.objectFields.push_back(nestedParam);
        }
      }
    }
  }
}

// Represents a FIRRTL annotation from TableGen
struct AnnotationDef {
  const Record *def;
  StringRef className;
  StringRef description;
  std::vector<ParameterDef> parameters;

  // Annotation type (detected from base class)
  enum class AnnotationType { NoTarget, SingleTarget, Ignored } annoType;

  // For SingleTargetAnnotation
  bool allowNonLocal = false;
  bool allowPortTargets = false;
  std::vector<StringRef>
      targets; // MLIR operation names (e.g., "WireOp", "FModuleOp")

  AnnotationDef(const Record *def) : def(def) {
    className = def->getValueAsString("className");
    description = def->getValueAsOptionalString("description").value_or("");

    // Detect annotation type from base class
    if (def->isSubClassOf("IgnoredAnnotation")) {
      annoType = AnnotationType::Ignored;
    } else if (def->isSubClassOf("NoTargetAnnotation")) {
      annoType = AnnotationType::NoTarget;
    } else if (def->isSubClassOf("SingleTargetAnnotation")) {
      annoType = AnnotationType::SingleTarget;
      allowNonLocal = def->getValueAsBit("allowNonLocal");
      allowPortTargets = def->getValueAsBit("allowPortTargets");

      // Get target types and their operation names
      const auto *targetsInit = def->getValueAsListInit("targets");
      for (auto *val : targetsInit->getElements()) {
        if (auto *defInit = dyn_cast<DefInit>(val)) {
          // Get the TargetType record and read its operation name
          const Record *targetTypeRec = defInit->getDef();
          StringRef opName = targetTypeRec->getValueAsString("operation");
          if (!opName.empty()) {
            targets.push_back(opName);
          }
        }
      }
    } else {
      // Unknown annotation type - default to NoTarget
      annoType = AnnotationType::NoTarget;
    }

    // For SingleTargetAnnotation, automatically add 'target' parameter first
    if (annoType == AnnotationType::SingleTarget) {
      ParameterDef targetParam;
      targetParam.name = "target";
      targetParam.type = "target";
      targetParam.description = "Target of the annotation";
      targetParam.required = true;
      parameters.push_back(targetParam);
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
        ParameterDef param;
        parseParameterFromRecord(paramTypeRec, paramName, param);
        parameters.push_back(param);
      }
    }
  }
};

} // namespace

static bool emitMarkdownDocs(const RecordKeeper &records, raw_ostream &os) {
  os << "<!-- Autogenerated by circt-tblgen; don't manually edit -->\n\n";
  os << "# FIRRTL Annotations\n\n";
  os << "This document describes the annotations supported by CIRCT's FIRRTL "
        "compiler.\n\n";

  // Collect all annotations (from all base classes)
  std::vector<AnnotationDef> annotations;
  for (const auto *def : records.getAllDerivedDefinitions("Annotation"))
    annotations.emplace_back(def);

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
    std::function<void(const ParameterDef &, StringRef)> printParameter;
    printParameter = [&](const ParameterDef &param, StringRef prefix) {
      os << "| " << prefix << param.name << " | " << param.type << " | ";

      if (!param.description.empty())
        os << param.description;

      // Add allowed values if present
      if (!param.allowedValues.empty()) {
        os << " (";
        bool first = true;
        for (auto val : param.allowedValues) {
          if (!first)
            os << ", ";
          os << "`" << val << "`";
          first = false;
        }
        os << ")";
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
      if (param.type == "object" && !param.objectFields.empty()) {
        std::string nestedPrefix = (prefix + param.name + ".").str();
        for (const auto &field : param.objectFields) {
          printParameter(field, nestedPrefix);
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
      raw_indented_ostream(os).printReindented(anno.description.rtrim());
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
  std::vector<AnnotationDef> annotations;
  for (const auto *def : records.getAllDerivedDefinitions("Annotation")) {
    annotations.emplace_back(def);
  }

  // Sort by className for consistent output
  llvm::sort(annotations, [](const AnnotationDef &a, const AnnotationDef &b) {
    return a.className < b.className;
  });

  // Emit header comment
  os << "//===- FIRRTLAnnotationDetails.h.inc - Annotation Details -------*- "
        "C++ "
        "-*-===//\n";
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

  // Helper function to convert className to a valid C++ identifier
  auto makeIdentifier = [](StringRef className) -> std::string {
    std::string result;

    // Remove trailing '$' if present
    StringRef cleanName = className;
    if (cleanName.ends_with("$"))
      cleanName = cleanName.drop_back();

    // Take the last component after the last '.' or '$'
    size_t lastDot = cleanName.rfind('.');
    size_t lastDollar = cleanName.rfind('$');
    size_t start = 0;

    // Find the position of the last separator
    if (lastDot != StringRef::npos && lastDollar != StringRef::npos)
      start = std::max(lastDot, lastDollar) + 1;
    else if (lastDot != StringRef::npos)
      start = lastDot + 1;
    else if (lastDollar != StringRef::npos)
      start = lastDollar + 1;

    StringRef baseName = cleanName.substr(start);
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
  };

  // Emit all annotations
  for (const auto &anno : annotations) {
    std::string varName = makeIdentifier(anno.className);

    // Add comment if there's a description
    if (!anno.description.empty()) {
      // Extract first line of description as a brief comment
      StringRef desc = anno.description.trim();
      size_t firstNewline = desc.find('\n');
      StringRef firstLine =
          firstNewline == StringRef::npos ? desc : desc.substr(0, firstNewline);
      os << "/// " << firstLine.trim() << "\n";
    }

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
  std::vector<AnnotationDef> annotations;
  for (const auto *def : records.getAllDerivedDefinitions("Annotation"))
    annotations.emplace_back(def);

  // Emit header comment
  os << "//===- FIRRTLAnnotationRecords.inc - Annotation Records --------*- "
        "C++ "
        "-*-===//\n";
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

  // Helper to convert className to variable name
  // This MUST match the logic in emitAnnotationDetails() to ensure consistency
  auto makeVarName = [](StringRef className) -> std::string {
    std::string result;

    // Remove trailing '$' if present
    StringRef cleanName = className;
    if (cleanName.ends_with("$"))
      cleanName = cleanName.drop_back();

    // Take the last component after the last '.'
    size_t lastDot = cleanName.rfind('.');
    size_t lastDollar = cleanName.rfind('$');
    size_t start = 0;

    // Find the position of the last separator
    if (lastDot != StringRef::npos && lastDollar != StringRef::npos)
      start = std::max(lastDot, lastDollar) + 1;
    else if (lastDot != StringRef::npos)
      start = lastDot + 1;
    else if (lastDollar != StringRef::npos)
      start = lastDollar + 1;

    StringRef baseName = cleanName.substr(start);
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
  };

  // Emit annotation records
  for (const auto &anno : annotations) {
    std::string varName = makeVarName(anno.className);

    // Determine resolver based on annotation type
    std::string resolver;
    if (anno.annoType == AnnotationDef::AnnotationType::NoTarget ||
        anno.annoType == AnnotationDef::AnnotationType::Ignored)
      resolver = "noResolve";
    else
      resolver = "stdResolve";

    // Determine applier - auto-generate from annotation metadata
    std::string applier;
    StringRef customHandler = anno.def->getValueAsString("customHandler");

    // If customHandler is specified, use it directly (manual override)
    if (!customHandler.empty()) {
      applier = customHandler.str();
    } else if (anno.annoType == AnnotationDef::AnnotationType::Ignored) {
      // IgnoredAnnotation - always drop
      applier = "drop";
    } else if (anno.annoType == AnnotationDef::AnnotationType::NoTarget) {
      // NoTargetAnnotation - apply to CircuitOp
      applier = "applyWithoutTarget<false, CircuitOp>";
    } else if (anno.annoType == AnnotationDef::AnnotationType::SingleTarget) {
      // SingleTargetAnnotation - auto-generate from metadata
      // Build the handler based on targets, allowNonLocal, allowPortTargets

      if (!anno.targets.empty()) {
        // Specific target types - use the operation names directly
        std::string opTypes;
        for (size_t i = 0; i < anno.targets.size(); ++i) {
          if (i > 0)
            opTypes += ", ";
          opTypes += anno.targets[i].str();
        }

        // Build template arguments: <allowNonLocal, allowPortTargets,
        // OpTypes...>
        applier = "applyWithoutTarget<";
        applier += anno.allowNonLocal ? "true" : "false";
        applier += ", ";
        applier += anno.allowPortTargets ? "true" : "false";
        applier += ", ";
        applier += opTypes;
        applier += ">";
      } else {
        // No specific target types - any named target is allowed
        applier = "applyWithoutTarget<";
        applier += anno.allowNonLocal ? "true" : "false";
        applier += ">";
      }
    } else {
      // Fallback - should not happen
      PrintWarning(anno.def->getLoc(),
                   "Unknown annotation type for '" + anno.className + "'");
      applier = "drop";
    }

    os << "  {" << varName << ", {" << resolver << ", " << applier << "}},\n";
  }

  os << "\n#undef GET_ANNOTATION_RECORD_LIST\n";
  os << "#endif // GET_ANNOTATION_RECORD_LIST\n";

  return false;
}

static GenRegistration genRecords("firrtl-annotation-records",
                                  "Generate annotation handler records",
                                  emitAnnotationRecords);
