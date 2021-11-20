//===- EmitOMIR.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the EmitOMIR pass.
//
//===----------------------------------------------------------------------===//

#include "AnnotationDetails.h"
#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/InstanceGraph.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"

#define DEBUG_TYPE "omir"

using namespace circt;
using namespace firrtl;
using mlir::LocationAttr;
using mlir::UnitAttr;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace {
/// Information concerning a tracker in the IR.
struct Tracker {
  /// The unique ID of this tracker.
  IntegerAttr id;
  /// The operation onto which this tracker was annotated.
  Operation *op;
  /// If this tracker is non-local, this is the corresponding anchor.
  NonLocalAnchor nla;
};

class EmitOMIRPass : public EmitOMIRBase<EmitOMIRPass> {
public:
  using EmitOMIRBase::outputFilename;

private:
  void runOnOperation() override;
  void makeTrackerAbsolute(Tracker &tracker);

  void emitSourceInfo(Location input, SmallString<64> &into);
  void emitOMNode(Attribute node, llvm::json::OStream &jsonStream);
  void emitOMField(StringAttr fieldName, DictionaryAttr field,
                   llvm::json::OStream &jsonStream);
  void emitOptionalRTLPorts(DictionaryAttr node,
                            llvm::json::OStream &jsonStream);
  void emitValue(Attribute node, llvm::json::OStream &jsonStream);
  void emitTrackedTarget(DictionaryAttr node, llvm::json::OStream &jsonStream);

  SmallString<8> addSymbolImpl(Attribute symbol) {
    unsigned id;
    auto it = symbolIndices.find(symbol);
    if (it != symbolIndices.end()) {
      id = it->second;
    } else {
      id = symbols.size();
      symbols.push_back(symbol);
      symbolIndices.insert({symbol, id});
    }
    SmallString<8> str;
    ("{{" + Twine(id) + "}}").toVector(str);
    return str;
  }
  SmallString<8> addSymbol(hw::InnerRefAttr symbol) {
    return addSymbolImpl(symbol);
  }
  SmallString<8> addSymbol(FlatSymbolRefAttr symbol) {
    return addSymbolImpl(symbol);
  }
  SmallString<8> addSymbol(StringAttr symbolName) {
    return addSymbol(FlatSymbolRefAttr::get(symbolName));
  }
  SmallString<8> addSymbol(Operation *op) {
    return addSymbol(SymbolTable::getSymbolName(op));
  }

  /// Returns an operation's `inner_sym`, adding one if necessary.
  StringAttr getOrAddInnerSym(Operation *op);
  /// Returns a port's `inner_sym`, adding one if necessary.
  StringAttr getOrAddInnerSym(FModuleLike module, size_t portIdx);
  /// Obtain an inner reference to an operation, possibly adding an `inner_sym`
  /// to that operation.
  hw::InnerRefAttr getInnerRefTo(Operation *op);
  /// Obtain an inner reference to a module port, possibly adding an `inner_sym`
  /// to that port.
  hw::InnerRefAttr getInnerRefTo(FModuleLike module, size_t portIdx);

  /// Get the cached namespace for a module.
  ModuleNamespace &getModuleNamespace(FModuleLike module) {
    auto it = moduleNamespaces.find(module);
    if (it != moduleNamespaces.end())
      return it->second;
    return moduleNamespaces.insert({module, ModuleNamespace(module)})
        .first->second;
  }

  /// Whether any errors have occurred in the current `runOnOperation`.
  bool anyFailures;
  /// Analyses for the current operation; only valid within `runOnOperation`.
  SymbolTable *symtbl;
  CircuitNamespace *circuitNamespace;
  InstancePathCache *instancePaths;
  /// OMIR target trackers gathered in the current operation, by tracker ID.
  DenseMap<Attribute, Tracker> trackers;
  /// The list of symbols to be interpolated in the verbatim JSON. This gets
  /// populated as the JSON is constructed and module and instance names are
  /// collected.
  SmallVector<Attribute> symbols;
  SmallDenseMap<Attribute, unsigned> symbolIndices;
  /// Temporary `firrtl.nla` operations to be deleted at the end of the pass.
  SmallVector<NonLocalAnchor> removeTempNLAs;
  DenseMap<Operation *, ModuleNamespace> moduleNamespaces;
};
} // namespace

/// Check if an `OMNode` is an `OMSRAM` and requires special treatment of its
/// instance path field. This returns the ID of the tracker stored in the
/// `instancePath` or `finalPath` field if the node has an array field `omType`
/// that contains a `OMString:OMSRAM` entry.
static IntegerAttr isOMSRAM(Attribute &node) {
  auto dict = node.dyn_cast<DictionaryAttr>();
  if (!dict)
    return {};
  auto idAttr = dict.getAs<StringAttr>("id");
  if (!idAttr)
    return {};
  IntegerAttr id;
  if (auto infoAttr = dict.getAs<DictionaryAttr>("fields")) {
    auto finalPath = infoAttr.getAs<DictionaryAttr>("finalPath");
    // The following is used prior to an upstream bump in Chisel.
    if (!finalPath)
      finalPath = infoAttr.getAs<DictionaryAttr>("instancePath");
    if (finalPath)
      if (auto v = finalPath.getAs<DictionaryAttr>("value"))
        if (v.getAs<UnitAttr>("omir.tracker"))
          id = v.getAs<IntegerAttr>("id");
    if (auto omTy = infoAttr.getAs<DictionaryAttr>("omType"))
      if (auto valueArr = omTy.getAs<ArrayAttr>("value"))
        for (auto attr : valueArr)
          if (auto str = attr.dyn_cast<StringAttr>())
            if (str.getValue().equals("OMString:OMSRAM"))
              return id;
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

void EmitOMIRPass::runOnOperation() {
  MLIRContext *context = &getContext();
  anyFailures = false;
  symtbl = nullptr;
  circuitNamespace = nullptr;
  instancePaths = nullptr;
  trackers.clear();
  symbols.clear();
  symbolIndices.clear();
  removeTempNLAs.clear();
  moduleNamespaces.clear();
  CircuitOp circuitOp = getOperation();

  // Gather the relevant annotations from the circuit. On the one hand these are
  // all the actual `OMIRAnnotation`s that need processing and emission, as well
  // as an optional `OMIRFileAnnotation` that overrides the default OMIR output
  // file. Also while we're at it, keep track of all OMIR nodes that qualify as
  // an SRAM and that require their trackers to be turned into NLAs starting at
  // the root of the hierarchy.
  SmallVector<ArrayRef<Attribute>> annoNodes;
  DenseSet<Attribute> sramIDs;
  Optional<StringRef> outputFilename = {};

  AnnotationSet::removeAnnotations(circuitOp, [&](Annotation anno) {
    if (anno.isClass(omirFileAnnoClass)) {
      auto pathAttr = anno.getMember<StringAttr>("filename");
      if (!pathAttr) {
        circuitOp.emitError(omirFileAnnoClass)
            << " annotation missing `filename` string attribute";
        anyFailures = true;
        return true;
      }
      LLVM_DEBUG(llvm::dbgs() << "- OMIR path: " << pathAttr << "\n");
      outputFilename = pathAttr.getValue();
      return true;
    }
    if (anno.isClass(omirAnnoClass)) {
      auto nodesAttr = anno.getMember<ArrayAttr>("nodes");
      if (!nodesAttr) {
        circuitOp.emitError(omirAnnoClass)
            << " annotation missing `nodes` array attribute";
        anyFailures = true;
        return true;
      }
      LLVM_DEBUG(llvm::dbgs() << "- OMIR: " << nodesAttr << "\n");
      annoNodes.push_back(nodesAttr.getValue());
      for (auto node : nodesAttr) {
        if (auto id = isOMSRAM(node)) {
          LLVM_DEBUG(llvm::dbgs() << "  - is SRAM with tracker " << id << "\n");
          sramIDs.insert(id);
        }
      }
      return true;
    }
    return false;
  });
  if (anyFailures)
    return signalPassFailure();

  // Establish some of the analyses we need throughout the pass.
  SymbolTable currentSymtbl(circuitOp);
  CircuitNamespace currentCircuitNamespace(circuitOp);
  InstancePathCache currentInstancePaths(getAnalysis<InstanceGraph>());
  symtbl = &currentSymtbl;
  circuitNamespace = &currentCircuitNamespace;
  instancePaths = &currentInstancePaths;

  // Traverse the IR and collect all tracker annotations that were previously
  // scattered into the circuit.
  circuitOp.walk([&](Operation *op) {
    AnnotationSet::removeAnnotations(op, [&](Annotation anno) {
      if (!anno.isClass(omirTrackerAnnoClass))
        return false;
      Tracker tracker;
      tracker.op = op;
      tracker.id = anno.getMember<IntegerAttr>("id");
      if (!tracker.id) {
        op->emitError(omirTrackerAnnoClass)
            << " annotation missing `id` integer attribute";
        anyFailures = true;
        return true;
      }
      if (auto nlaSym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal"))
        tracker.nla =
            dyn_cast_or_null<NonLocalAnchor>(symtbl->lookup(nlaSym.getAttr()));
      if (sramIDs.erase(tracker.id))
        makeTrackerAbsolute(tracker);
      trackers.insert({tracker.id, tracker});
      return true;
    });
  });

  // If an OMIR output filename has been specified as a pass parameter, override
  // whatever the annotations have configured. If neither are specified we just
  // bail.
  if (!this->outputFilename.empty())
    outputFilename = this->outputFilename;
  if (!outputFilename) {
    LLVM_DEBUG(llvm::dbgs() << "Not emitting OMIR because no annotation or "
                               "pass parameter specified an output file\n");
    markAllAnalysesPreserved();
    return;
  }

  // Build the output JSON.
  std::string jsonBuffer;
  llvm::raw_string_ostream jsonOs(jsonBuffer);
  llvm::json::OStream json(jsonOs, 2);
  json.array([&] {
    for (auto nodes : annoNodes) {
      for (auto node : nodes) {
        emitOMNode(node, json);
        if (anyFailures)
          return;
      }
    }
  });
  if (anyFailures)
    return signalPassFailure();

  // Delete the temporary NLAs. This requires us to visit all the nodes along
  // the NLA's path and remove `circt.nonlocal` annotations referring to the
  // NLA.
  for (auto nla : removeTempNLAs) {
    LLVM_DEBUG(llvm::dbgs() << "Removing " << nla << "\n");
    for (auto modName : nla.modpath().getAsRange<FlatSymbolRefAttr>()) {
      Operation *mod = symtbl->lookup(modName.getValue());
      mod->walk([&](InstanceOp instOp) {
        AnnotationSet::removeAnnotations(instOp, [&](Annotation anno) {
          auto match =
              anno.isClass("circt.nonlocal") &&
              anno.getMember<FlatSymbolRefAttr>("circt.nonlocal").getAttr() ==
                  nla.sym_nameAttr();
          if (match)
            LLVM_DEBUG(llvm::dbgs()
                       << "- Removing " << anno.getDict() << " from " << modName
                       << "." << instOp.name() << "\n");
          return match;
        });
      });
    }
    nla->erase();
  }
  removeTempNLAs.clear();

  // Emit the OMIR JSON as a verbatim op.
  auto builder = OpBuilder(circuitOp);
  builder.setInsertionPointAfter(circuitOp);
  auto verbatimOp =
      builder.create<sv::VerbatimOp>(builder.getUnknownLoc(), jsonBuffer);
  auto fileAttr = hw::OutputFileAttr::getFromFilename(
      context, *outputFilename, /*excludeFromFilelist=*/true);
  verbatimOp->setAttr("output_file", fileAttr);
  verbatimOp.symbolsAttr(ArrayAttr::get(context, symbols));
}

/// Make a tracker absolute by adding an NLA to it which starts at the root
/// module of the circuit. Generates an error if any module along the path is
/// instantiated multiple times.
void EmitOMIRPass::makeTrackerAbsolute(Tracker &tracker) {
  auto *context = &getContext();
  auto builder = OpBuilder::atBlockBegin(getOperation().getBody());

  // Pick a name for the NLA that doesn't collide with anything.
  auto opName = tracker.op->getAttrOfType<StringAttr>("name");
  auto nlaName = circuitNamespace->newName("omir_nla_" + opName.getValue());

  // Assemble the NLA annotation to be put on all the operations participating
  // in the path.
  auto nlaAttr = builder.getDictionaryAttr({
      builder.getNamedAttr("circt.nonlocal",
                           FlatSymbolRefAttr::get(context, nlaName)),
      builder.getNamedAttr("class", StringAttr::get(context, "circt.nonlocal")),
  });

  // Get all the paths instantiating this module.
  auto mod = tracker.op->getParentOfType<FModuleOp>();
  auto paths = instancePaths->getAbsolutePaths(mod);
  if (paths.empty()) {
    tracker.op->emitError("OMIR node targets uninstantiated component `")
        << opName.getValue() << "`";
    anyFailures = true;
    return;
  }
  if (paths.size() > 1) {
    auto diag = tracker.op->emitError("OMIR node targets ambiguous component `")
                << opName.getValue() << "`";
    diag.attachNote(tracker.op->getLoc())
        << "may refer to the following paths:";
    for (auto path : paths)
      formatInstancePath(diag.attachNote(tracker.op->getLoc()) << "- ", path);
    anyFailures = true;
    return;
  }

  // Assemble the module and name path for the NLA. Also attach an NLA reference
  // annotation to each instance participating in the path.
  SmallVector<Attribute> modpath, namepath;
  auto addToPath = [&](Operation *op, StringAttr name) {
    AnnotationSet annos(op);
    annos.addAnnotations(nlaAttr);
    annos.applyToOperation(op);
    modpath.push_back(FlatSymbolRefAttr::get(op->getParentOfType<FModuleOp>()));
    namepath.push_back(name);
  };
  for (InstanceOp inst : paths[0])
    addToPath(inst, inst.nameAttr());
  addToPath(tracker.op, opName);

  // Add the NLA to the tracker and mark it to be deleted later.
  tracker.nla = builder.create<NonLocalAnchor>(
      builder.getUnknownLoc(), builder.getStringAttr(nlaName),
      builder.getArrayAttr(modpath), builder.getArrayAttr(namepath));
  removeTempNLAs.push_back(tracker.nla);
}

/// Emit a source locator into a string, for inclusion in the `info` field of
/// `OMNode` and `OMField`.
void EmitOMIRPass::emitSourceInfo(Location input, SmallString<64> &into) {
  into.clear();
  input->walk([&](Location loc) {
    if (FileLineColLoc fileLoc = loc.dyn_cast<FileLineColLoc>()) {
      into.append(into.empty() ? "@[" : " ");
      (Twine(fileLoc.getFilename()) + " " + Twine(fileLoc.getLine()) + ":" +
       Twine(fileLoc.getColumn()))
          .toVector(into);
    }
    return WalkResult::advance();
  });
  if (!into.empty())
    into.append("]");
  else
    into.append("UnlocatableSourceInfo");
}

/// Emit an entire `OMNode` as JSON.
void EmitOMIRPass::emitOMNode(Attribute node, llvm::json::OStream &jsonStream) {
  auto dict = node.dyn_cast<DictionaryAttr>();
  if (!dict) {
    getOperation()
            .emitError("OMNode must be a dictionary")
            .attachNote(getOperation().getLoc())
        << node;
    anyFailures = true;
    return;
  }

  // Extract the `info` field and serialize the location.
  SmallString<64> info;
  if (auto infoAttr = dict.getAs<LocationAttr>("info"))
    emitSourceInfo(infoAttr, info);
  if (anyFailures)
    return;

  // Extract the `id` field.
  auto idAttr = dict.getAs<StringAttr>("id");
  if (!idAttr) {
    getOperation()
            .emitError("OMNode missing `id` string field")
            .attachNote(getOperation().getLoc())
        << dict;
    anyFailures = true;
    return;
  }

  // Extract and order the fields of this node.
  SmallVector<std::tuple<unsigned, StringAttr, DictionaryAttr>> orderedFields;
  auto fieldsDict = dict.getAs<DictionaryAttr>("fields");
  if (fieldsDict) {
    for (auto nameAndField : fieldsDict.getValue()) {
      auto fieldDict = nameAndField.getValue().dyn_cast<DictionaryAttr>();
      if (!fieldDict) {
        getOperation()
                .emitError("OMField must be a dictionary")
                .attachNote(getOperation().getLoc())
            << nameAndField.getValue();
        anyFailures = true;
        return;
      }

      unsigned index = 0;
      if (auto indexAttr = fieldDict.getAs<IntegerAttr>("index"))
        index = indexAttr.getValue().getLimitedValue();

      orderedFields.push_back({index, nameAndField.getName(), fieldDict});
    }
    llvm::sort(orderedFields,
               [](auto a, auto b) { return std::get<0>(a) < std::get<0>(b); });
  }

  jsonStream.object([&] {
    jsonStream.attribute("info", info);
    jsonStream.attribute("id", idAttr.getValue());
    jsonStream.attributeArray("fields", [&] {
      for (auto &orderedField : orderedFields) {
        emitOMField(std::get<1>(orderedField), std::get<2>(orderedField),
                    jsonStream);
        if (anyFailures)
          return;
      }
      if (auto node = fieldsDict.getAs<DictionaryAttr>("containingModule"))
        if (auto value = node.getAs<DictionaryAttr>("value"))
          emitOptionalRTLPorts(value, jsonStream);
    });
  });
}

/// Emit a single `OMField` as JSON. This expects the field's name to be
/// provided from the outside, for example as the field name that this attribute
/// has in the surrounding dictionary.
void EmitOMIRPass::emitOMField(StringAttr fieldName, DictionaryAttr field,
                               llvm::json::OStream &jsonStream) {
  // Extract the `info` field and serialize the location.
  auto infoAttr = field.getAs<LocationAttr>("info");
  SmallString<64> info;
  if (infoAttr)
    emitSourceInfo(infoAttr, info);
  if (anyFailures)
    return;

  jsonStream.object([&] {
    jsonStream.attribute("info", info);
    jsonStream.attribute("name", fieldName.strref());
    jsonStream.attributeBegin("value");
    emitValue(field.get("value"), jsonStream);
    jsonStream.attributeEnd();
  });
}

// If the given `node` refers to a valid tracker in the IR, gather the
// additional port metadata of the module it refers to. Then emit this port
// metadata as a `ports` array field for the surrounding `OMNode`.
void EmitOMIRPass::emitOptionalRTLPorts(DictionaryAttr node,
                                        llvm::json::OStream &jsonStream) {
  // First make sure we actually have a valid tracker. If not, just silently
  // abort and don't emit any port metadata.
  auto idAttr = node.getAs<IntegerAttr>("id");
  auto trackerIt = trackers.find(idAttr);
  if (!idAttr || !node.getAs<UnitAttr>("omir.tracker") ||
      trackerIt == trackers.end())
    return;
  auto tracker = trackerIt->second;

  // Lookup the module the tracker refers to. If it points at something *within*
  // a module, go dig up the surrounding module. This is roughly what
  // `Target.referringModule(...)` does on the Scala side.
  auto module = dyn_cast<FModuleLike>(tracker.op);
  if (!module)
    module = tracker.op->getParentOfType<FModuleLike>();
  if (!module) {
    LLVM_DEBUG(llvm::dbgs() << "Not emitting RTL ports since tracked operation "
                               "does not have a FModuleLike parent: "
                            << *tracker.op << "\n");
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "Emitting RTL ports for module `"
                          << module.moduleName() << "`\n");

  // Emit the JSON.
  SmallString<64> buf;
  jsonStream.object([&] {
    buf.clear();
    emitSourceInfo(module.getLoc(), buf);
    jsonStream.attribute("info", buf);
    jsonStream.attribute("name", "ports");
    jsonStream.attributeArray("value", [&] {
      for (auto port : llvm::enumerate(module.getPorts())) {
        if (port.value().type.getBitWidthOrSentinel() == 0)
          continue;
        jsonStream.object([&] {
          // Emit the `ref` field.
          buf.assign("OMDontTouchedReferenceTarget:~");
          buf.append(getOperation().name());
          buf.push_back('|');
          buf.append(addSymbol(module));
          buf.push_back('>');
          buf.append(addSymbol(getInnerRefTo(module, port.index())));
          jsonStream.attribute("ref", buf);

          // Emit the `direction` field.
          buf.assign("OMString:");
          buf.append(port.value().isOutput() ? "Output" : "Input");
          jsonStream.attribute("direction", buf);

          // Emit the `width` field.
          buf.assign("OMBigInt:");
          Twine(port.value().type.getBitWidthOrSentinel()).toVector(buf);
          jsonStream.attribute("width", buf);
        });
      }
    });
  });
}

void EmitOMIRPass::emitValue(Attribute node, llvm::json::OStream &jsonStream) {
  // Handle the null case.
  if (!node || node.isa<UnitAttr>())
    return jsonStream.value(nullptr);

  // Handle the trivial cases where the OMIR serialization simply uses the
  // builtin JSON types.
  if (auto attr = node.dyn_cast<BoolAttr>())
    return jsonStream.value(attr.getValue()); // OMBoolean
  if (auto attr = node.dyn_cast<IntegerAttr>()) {
    // CAVEAT: We expect these integers to come from an OMIR file that is
    // initially read in from JSON, where they are i32 or i64, so this should
    // yield a valid value. However, a user could cook up an arbitrary precision
    // integer attr in MLIR input and then subtly break the JSON spec.
    SmallString<16> val;
    attr.getValue().toStringSigned(val);
    return jsonStream.rawValue(val); // OMInt
  }
  if (auto attr = node.dyn_cast<FloatAttr>()) {
    // CAVEAT: We expect these floats to come from an OMIR file that is
    // initially read in from JSON, where they are f32 or f64, so this should
    // yield a valid value. However, a user could cook up an arbitrary precision
    // float attr in MLIR input and then subtly break the JSON spec.
    SmallString<16> val;
    attr.getValue().toString(val);
    return jsonStream.rawValue(val); // OMDouble
  }

  // Handle aggregate types.
  if (auto attr = node.dyn_cast<ArrayAttr>()) {
    jsonStream.array([&] {
      for (auto element : attr.getValue()) {
        emitValue(element, jsonStream);
        if (anyFailures)
          return;
      }
    });
    return;
  }
  if (auto attr = node.dyn_cast<DictionaryAttr>()) {
    // Handle targets that have a corresponding tracker annotation in the IR.
    if (attr.getAs<UnitAttr>("omir.tracker"))
      return emitTrackedTarget(attr, jsonStream);

    // Handle regular dictionaries.
    jsonStream.object([&] {
      for (auto field : attr.getValue()) {
        jsonStream.attributeBegin(field.getName());
        emitValue(field.getValue(), jsonStream);
        jsonStream.attributeEnd();
        if (anyFailures)
          return;
      }
    });
    return;
  }

  // The remaining types are all simple string-encoded pass-through cases.
  if (auto attr = node.dyn_cast<StringAttr>()) {
    StringRef val = attr.getValue();
    if (isOMIRStringEncodedPassthrough(val.split(":").first))
      return jsonStream.value(val);
  }

  // If we get here, we don't know how to serialize the given MLIR attribute as
  // a OMIR value.
  jsonStream.value("<unsupported value>");
  getOperation().emitError("unsupported attribute for OMIR serialization: `")
      << node << "`";
  anyFailures = true;
}

void EmitOMIRPass::emitTrackedTarget(DictionaryAttr node,
                                     llvm::json::OStream &jsonStream) {
  // Extract the `id` field.
  auto idAttr = node.getAs<IntegerAttr>("id");
  if (!idAttr) {
    getOperation()
            .emitError("tracked OMIR target missing `id` string field")
            .attachNote(getOperation().getLoc())
        << node;
    anyFailures = true;
    return jsonStream.value("<error>");
  }

  // Extract the `type` field.
  auto typeAttr = node.getAs<StringAttr>("type");
  if (!typeAttr) {
    getOperation()
            .emitError("tracked OMIR target missing `type` string field")
            .attachNote(getOperation().getLoc())
        << node;
    anyFailures = true;
    return jsonStream.value("<error>");
  }
  StringRef type = typeAttr.getValue();

  // Find the tracker for this target, and handle the case where the tracker has
  // been deleted.
  auto trackerIt = trackers.find(idAttr);
  if (trackerIt == trackers.end()) {
    // Some of the target types indicate removal of the target through an
    // `OMDeleted` node.
    if (type == "OMReferenceTarget" || type == "OMMemberReferenceTarget" ||
        type == "OMMemberInstanceTarget")
      return jsonStream.value("OMDeleted:");

    // The remaining types produce an error upon removal of the target.
    auto diag = getOperation().emitError("tracked OMIR target of type `")
                << type << "` was deleted";
    diag.attachNote(getOperation().getLoc())
        << "`" << type << "` should never be deleted";
    if (auto path = node.getAs<StringAttr>("path"))
      diag.attachNote(getOperation().getLoc())
          << "original path: `" << path.getValue() << "`";
    anyFailures = true;
    return jsonStream.value("<error>");
  }
  auto tracker = trackerIt->second;

  // In case this is an `OMMemberTarget`, handle the case where the component
  // used to be a "reference target" (wire, register, memory, node) when the
  // OMIR was read in, but has been change to an "instance target" during the
  // execution of the compiler. This mainly occurs during mapping of
  // `firrtl.mem` operations to a corresponding `firrtl.instance`.
  if (type == "OMMemberReferenceTarget" && isa<InstanceOp, MemOp>(tracker.op))
    type = "OMMemberInstanceTarget";

  // Serialize the target circuit first.
  SmallString<64> target(type);
  target.append(":~");
  target.append(getOperation().name());
  target.push_back('|');

  // Serialize the local or non-local module/instance hierarchy path.
  if (tracker.nla) {
    bool notFirst = false;
    hw::InnerRefAttr instName;
    for (auto modAndName : llvm::zip(tracker.nla.modpath().getValue(),
                                     tracker.nla.namepath().getValue())) {
      auto symAttr = std::get<0>(modAndName).cast<FlatSymbolRefAttr>();
      auto nameAttr = std::get<1>(modAndName).cast<StringAttr>();
      Operation *module = symtbl->lookup(symAttr.getValue());
      assert(module);
      if (notFirst)
        target.push_back('/');
      notFirst = true;
      if (instName) {
        target.append(addSymbol(instName));
        target.push_back(':');
      }
      target.append(addSymbol(module));

      // Find an instance with the given name in this module. Ensure it has a
      // symbol that we can refer to.
      instName = {};
      module->walk([&](InstanceOp instOp) {
        if (instOp.nameAttr() != nameAttr)
          return;
        LLVM_DEBUG(llvm::dbgs()
                   << "Marking NLA-participating instance " << nameAttr
                   << " in module " << symAttr << " as dont-touch\n");
        AnnotationSet::addDontTouch(instOp);
        instName = getInnerRefTo(instOp);
      });
    }
  } else {
    FModuleOp module = dyn_cast<FModuleOp>(tracker.op);
    if (!module)
      module = tracker.op->getParentOfType<FModuleOp>();
    assert(module);
    target.append(addSymbol(module));
  }

  // Serialize any potential component *inside* the module that this target may
  // specifically refer to.
  hw::InnerRefAttr componentName;
  if (isa<WireOp, RegOp, RegResetOp, InstanceOp, NodeOp, MemOp>(tracker.op)) {
    componentName = getInnerRefTo(tracker.op);
    AnnotationSet::addDontTouch(tracker.op);
    LLVM_DEBUG(llvm::dbgs() << "Marking OMIR-targeted " << componentName
                            << " as dont-touch\n");
  } else if (!isa<FModuleOp>(tracker.op)) {
    tracker.op->emitError("invalid target for `") << type << "` OMIR";
    anyFailures = true;
    return jsonStream.value("<error>");
  }
  if (componentName) {
    // Check if the targeted component is going to be emitted as an instance.
    // This is trivially the case for `InstanceOp`s, but also for `MemOp`s that
    // get converted to an instance during lowering to HW dialect and generator
    // callout.
    [&] {
      if (type == "OMMemberInstanceTarget") {
        if (auto instOp = dyn_cast<InstanceOp>(tracker.op)) {
          target.push_back('/');
          target.append(addSymbol(componentName));
          target.push_back(':');
          target.append(addSymbol(instOp.moduleNameAttr()));
          return;
        }
        if (auto memOp = dyn_cast<MemOp>(tracker.op)) {
          target.push_back('/');
          target.append(addSymbol(componentName));
          target.push_back(':');
          target.append(memOp.getSummary().getFirMemoryName());
          return;
        }
      }
      target.push_back('>');
      target.append(addSymbol(componentName));
    }();
  }

  jsonStream.value(target);
}

StringAttr EmitOMIRPass::getOrAddInnerSym(Operation *op) {
  auto attr = op->getAttrOfType<StringAttr>("inner_sym");
  if (attr)
    return attr;
  auto module = op->getParentOfType<FModuleOp>();
  auto name = getModuleNamespace(module).newName("omir_sym");
  attr = StringAttr::get(op->getContext(), name);
  op->setAttr("inner_sym", attr);
  return attr;
}

StringAttr EmitOMIRPass::getOrAddInnerSym(FModuleLike module, size_t portIdx) {
  auto attr = module.getPortSymbolAttr(portIdx);
  if (attr && !attr.getValue().empty())
    return attr;
  auto name = getModuleNamespace(module).newName("omir_sym");
  attr = StringAttr::get(module.getContext(), name);
  module.setPortSymbolAttr(portIdx, attr);
  return attr;
}

hw::InnerRefAttr EmitOMIRPass::getInnerRefTo(Operation *op) {
  return hw::InnerRefAttr::get(
      SymbolTable::getSymbolName(op->getParentOfType<FModuleOp>()),
      getOrAddInnerSym(op));
}

hw::InnerRefAttr EmitOMIRPass::getInnerRefTo(FModuleLike module,
                                             size_t portIdx) {
  return hw::InnerRefAttr::get(SymbolTable::getSymbolName(module),
                               getOrAddInnerSym(module, portIdx));
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass>
circt::firrtl::createEmitOMIRPass(StringRef outputFilename) {
  auto pass = std::make_unique<EmitOMIRPass>();
  if (!outputFilename.empty())
    pass->outputFilename = outputFilename.str();
  return pass;
}
