//===- CreateSiFiveMetadata.cpp - Create various metadata -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CreateSiFiveMetadata pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_CREATESIFIVEMETADATA
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {

struct ObjectModelIR {
  ObjectModelIR(
      CircuitOp circtOp, FModuleOp dutMod, InstanceGraph &instanceGraph,
      DenseMap<Operation *, hw::InnerSymbolNamespace> &moduleNamespaces)
      : circtOp(circtOp), dutMod(dutMod),
        circtNamespace(CircuitNamespace(circtOp)),
        instancePathCache(InstancePathCache(instanceGraph)),
        moduleNamespaces(moduleNamespaces) {}

  // Add the tracker annotation to the op and get a PathOp to the op.
  PathOp createPathRef(Operation *op, hw::HierPathOp nla,
                       mlir::ImplicitLocOpBuilder &builderOM) {
    auto *context = op->getContext();

    NamedAttrList fields;
    auto id = DistinctAttr::create(UnitAttr::get(context));
    fields.append("id", id);
    fields.append("class", StringAttr::get(context, "circt.tracker"));
    if (nla)
      fields.append("circt.nonlocal", mlir::FlatSymbolRefAttr::get(nla));
    AnnotationSet annos(op);
    annos.addAnnotations(DictionaryAttr::get(context, fields));
    annos.applyToOperation(op);
    TargetKind kind = TargetKind::Reference;
    if (isa<InstanceOp, FModuleLike>(op))
      kind = TargetKind::Instance;

    // Create the path operation.
    return builderOM.create<PathOp>(kind, id);
  }

  // Create a ClassOp, with the specified fieldNames and fieldTypes as ports.
  // The output property is set from the input property port.
  ClassOp buildSimpleClassOp(OpBuilder &odsBuilder, Location loc, Twine name,
                             ArrayRef<StringRef> fieldNames,
                             ArrayRef<Type> fieldTypes) {
    SmallVector<PortInfo, 10> ports;
    for (auto [fieldName, fieldType] : llvm::zip(fieldNames, fieldTypes)) {
      ports.emplace_back(odsBuilder.getStringAttr(fieldName + "_in"), fieldType,
                         Direction::In);
      ports.emplace_back(odsBuilder.getStringAttr(fieldName), fieldType,
                         Direction::Out);
    }

    ClassOp classOp =
        odsBuilder.create<ClassOp>(loc, odsBuilder.getStringAttr(name), ports);
    Block *body = classOp.getBodyBlock();
    auto prevLoc = odsBuilder.saveInsertionPoint();
    odsBuilder.setInsertionPointToEnd(body);
    auto args = body->getArguments();
    for (unsigned i = 0, e = ports.size(); i != e; i += 2)
      odsBuilder.create<PropAssignOp>(loc, args[i + 1], args[i]);

    odsBuilder.restoreInsertionPoint(prevLoc);

    return classOp;
  }

  void createMemorySchema() {
    auto *context = circtOp.getContext();

    auto unknownLoc = mlir::UnknownLoc::get(context);
    auto builderOM = mlir::ImplicitLocOpBuilder::atBlockEnd(
        unknownLoc, circtOp.getBodyBlock());

    // Add all the properties of a memory as fields of the class.
    // The types must match exactly with the FMemModuleOp attribute type.

    mlir::Type extraPortsType[] = {
        StringType::get(context),  // name
        StringType::get(context),  // direction
        FIntegerType::get(context) // Width
    };
    StringRef extraPortFields[3] = {"name", "direction", "width"};

    extraPortsClass =
        buildSimpleClassOp(builderOM, unknownLoc, "ExtraPortsMemorySchema",
                           extraPortFields, extraPortsType);

    mlir::Type classFieldTypes[12] = {
        StringType::get(context),
        FIntegerType::get(context),
        FIntegerType::get(context),
        FIntegerType::get(context),
        FIntegerType::get(context),
        FIntegerType::get(context),
        FIntegerType::get(context),
        FIntegerType::get(context),
        FIntegerType::get(context),
        ListType::get(context, cast<PropertyType>(PathType::get(context))),
        BoolType::get(context),
        ListType::get(context,
                      cast<PropertyType>(detail::getInstanceTypeForClassLike(
                          extraPortsClass)))};

    memorySchemaClass =
        buildSimpleClassOp(builderOM, unknownLoc, "MemorySchema",
                           memoryParamNames, classFieldTypes);

    // Now create the class that will instantiate metadata class with all the
    // memories of the circt.
    SmallVector<PortInfo> mports;
    memoryMetadataClass = builderOM.create<ClassOp>(
        builderOM.getStringAttr("MemoryMetadata"), mports);
  }

  void createRetimeModulesSchema() {
    auto *context = circtOp.getContext();
    auto unknownLoc = mlir::UnknownLoc::get(context);
    auto builderOM = mlir::ImplicitLocOpBuilder::atBlockEnd(
        unknownLoc, circtOp.getBodyBlock());
    Type classFieldTypes[] = {StringType::get(context)};
    retimeModulesSchemaClass =
        buildSimpleClassOp(builderOM, unknownLoc, "RetimeModulesSchema",
                           retimeModulesParamNames, classFieldTypes);

    SmallVector<PortInfo> mports;
    retimeModulesMetadataClass = builderOM.create<ClassOp>(
        builderOM.getStringAttr("RetimeModulesMetadata"), mports);
  }

  void addRetimeModule(FModuleLike module) {
    if (!retimeModulesSchemaClass)
      createRetimeModulesSchema();
    auto builderOM = mlir::ImplicitLocOpBuilder::atBlockEnd(
        module->getLoc(), retimeModulesMetadataClass.getBodyBlock());

    // Create the path operation.
    auto modEntry =
        builderOM.create<StringConstantOp>(module.getModuleNameAttr());
    auto object = builderOM.create<ObjectOp>(retimeModulesSchemaClass,
                                             module.getModuleNameAttr());

    auto inPort = builderOM.create<ObjectSubfieldOp>(object, 0);
    builderOM.create<PropAssignOp>(inPort, modEntry);
    auto portIndex = retimeModulesMetadataClass.getNumPorts();
    SmallVector<std::pair<unsigned, PortInfo>> newPorts = {
        {portIndex,
         PortInfo(builderOM.getStringAttr(module.getName() + "_field"),
                  object.getType(), Direction::Out)}};
    retimeModulesMetadataClass.insertPorts(newPorts);
    auto blockarg = retimeModulesMetadataClass.getBodyBlock()->addArgument(
        object.getType(), module->getLoc());
    builderOM.create<PropAssignOp>(blockarg, object);
  }

  void addBlackBoxModulesSchema() {
    auto *context = circtOp.getContext();
    auto unknownLoc = mlir::UnknownLoc::get(context);
    auto builderOM = mlir::ImplicitLocOpBuilder::atBlockEnd(
        unknownLoc, circtOp.getBodyBlock());
    Type classFieldTypes[] = {StringType::get(context)};
    blackBoxModulesSchemaClass =
        buildSimpleClassOp(builderOM, unknownLoc, "SitestBlackBoxModulesSchema",
                           blackBoxModulesParamNames, classFieldTypes);
    SmallVector<PortInfo> mports;
    blackBoxMetadataClass = builderOM.create<ClassOp>(
        builderOM.getStringAttr("SitestBlackBoxMetadata"), mports);
  }

  void addBlackBoxModule(FExtModuleOp module) {
    if (!blackBoxModulesSchemaClass)
      addBlackBoxModulesSchema();
    StringRef defName = *module.getDefname();
    if (!blackboxModules.insert(defName).second)
      return;
    auto builderOM = mlir::ImplicitLocOpBuilder::atBlockEnd(
        module.getLoc(), blackBoxMetadataClass.getBodyBlock());
    auto modEntry = builderOM.create<StringConstantOp>(module.getDefnameAttr());
    auto object = builderOM.create<ObjectOp>(blackBoxModulesSchemaClass,
                                             module.getModuleNameAttr());

    auto inPort = builderOM.create<ObjectSubfieldOp>(object, 0);
    builderOM.create<PropAssignOp>(inPort, modEntry);
    auto portIndex = blackBoxMetadataClass.getNumPorts();
    SmallVector<std::pair<unsigned, PortInfo>> newPorts = {
        {portIndex,
         PortInfo(builderOM.getStringAttr(module.getName() + "_field"),
                  object.getType(), Direction::Out)}};
    blackBoxMetadataClass.insertPorts(newPorts);
    auto blockarg = blackBoxMetadataClass.getBodyBlock()->addArgument(
        object.getType(), module->getLoc());
    builderOM.create<PropAssignOp>(blockarg, object);
  }

  void addMemory(FMemModuleOp mem, bool inDut) {
    if (!memorySchemaClass)
      createMemorySchema();
    auto builderOM = mlir::ImplicitLocOpBuilder::atBlockEnd(
        mem.getLoc(), memoryMetadataClass.getBodyBlock());
    auto *context = builderOM.getContext();
    auto createConstField = [&](Attribute constVal) -> Value {
      if (auto boolConstant = dyn_cast_or_null<mlir::BoolAttr>(constVal))
        return builderOM.create<BoolConstantOp>(boolConstant);
      if (auto intConstant = dyn_cast_or_null<mlir::IntegerAttr>(constVal))
        return builderOM.create<FIntegerConstantOp>(intConstant);
      if (auto strConstant = dyn_cast_or_null<mlir::StringAttr>(constVal))
        return builderOM.create<StringConstantOp>(strConstant);
      return {};
    };
    auto nlaBuilder = OpBuilder::atBlockBegin(circtOp.getBodyBlock());

    auto memPaths = instancePathCache.getAbsolutePaths(mem);
    SmallVector<Value> memoryHierPaths;
    for (auto memPath : memPaths) {
      Operation *finalInst = memPath.leaf();
      SmallVector<Attribute> namepath;
      bool foundDut = dutMod == nullptr;
      for (auto inst : memPath) {
        if (!foundDut)
          if (inst->getParentOfType<FModuleOp>() == dutMod)
            foundDut = true;
        if (!foundDut)
          continue;

        namepath.emplace_back(firrtl::getInnerRefTo(
            inst, [&](auto mod) -> hw::InnerSymbolNamespace & {
              return getModuleNamespace(mod);
            }));
      }
      if (namepath.empty())
        continue;
      auto nla = nlaBuilder.create<hw::HierPathOp>(
          mem->getLoc(),
          nlaBuilder.getStringAttr(circtNamespace.newName("memNLA")),
          nlaBuilder.getArrayAttr(namepath));

      // Create the path operation.
      memoryHierPaths.emplace_back(createPathRef(finalInst, nla, builderOM));
    }
    auto hierpaths = builderOM.create<ListCreateOp>(
        ListType::get(context, cast<PropertyType>(PathType::get(context))),
        memoryHierPaths);
    SmallVector<Value> memFields;

    auto object = builderOM.create<ObjectOp>(memorySchemaClass, mem.getName());
    SmallVector<Value> extraPortsList;
    ClassType extraPortsType;
    for (auto attr : mem.getExtraPortsAttr()) {

      auto port = cast<DictionaryAttr>(attr);
      auto portName = createConstField(port.getAs<StringAttr>("name"));
      auto direction = createConstField(port.getAs<StringAttr>("direction"));
      auto width = createConstField(port.getAs<IntegerAttr>("width"));
      auto extraPortsObj =
          builderOM.create<ObjectOp>(extraPortsClass, "extraPorts");
      extraPortsType = extraPortsObj.getType();
      auto inPort = builderOM.create<ObjectSubfieldOp>(extraPortsObj, 0);
      builderOM.create<PropAssignOp>(inPort, portName);
      inPort = builderOM.create<ObjectSubfieldOp>(extraPortsObj, 2);
      builderOM.create<PropAssignOp>(inPort, direction);
      inPort = builderOM.create<ObjectSubfieldOp>(extraPortsObj, 4);
      builderOM.create<PropAssignOp>(inPort, width);
      extraPortsList.push_back(extraPortsObj);
    }
    auto extraPorts = builderOM.create<ListCreateOp>(
        memorySchemaClass.getPortType(22), extraPortsList);
    for (auto field : llvm::enumerate(memoryParamNames)) {
      auto propVal = createConstField(
          llvm::StringSwitch<TypedAttr>(field.value())
              .Case("name", builderOM.getStringAttr(mem.getName()))
              .Case("depth", mem.getDepthAttr())
              .Case("width", mem.getDataWidthAttr())
              .Case("maskBits", mem.getMaskBitsAttr())
              .Case("readPorts", mem.getNumReadPortsAttr())
              .Case("writePorts", mem.getNumWritePortsAttr())
              .Case("readwritePorts", mem.getNumReadWritePortsAttr())
              .Case("readLatency", mem.getReadLatencyAttr())
              .Case("writeLatency", mem.getWriteLatencyAttr())
              .Case("hierarchy", {})
              .Case("inDut", BoolAttr::get(context, inDut))
              .Case("extraPorts", {}));
      if (!propVal) {
        if (field.value() == "hierarchy")
          propVal = hierpaths;
        else
          propVal = extraPorts;
      }

      // The memory schema is a simple class, with input tied to output. The
      // arguments are ordered such that, port index i is the input that is tied
      // to i+1 which is the output.
      // The following `2*index` translates the index to the memory schema input
      // port number.
      auto inPort =
          builderOM.create<ObjectSubfieldOp>(object, 2 * field.index());
      builderOM.create<PropAssignOp>(inPort, propVal);
    }
    auto portIndex = memoryMetadataClass.getNumPorts();
    SmallVector<std::pair<unsigned, PortInfo>> newPorts = {
        {portIndex, PortInfo(builderOM.getStringAttr(mem.getName() + "_field"),
                             object.getType(), Direction::Out)}};
    memoryMetadataClass.insertPorts(newPorts);
    auto blockarg = memoryMetadataClass.getBodyBlock()->addArgument(
        object.getType(), mem->getLoc());
    builderOM.create<PropAssignOp>(blockarg, object);
  }

  ObjectOp instantiateSifiveMetadata(FModuleOp topMod) {
    if (!blackBoxMetadataClass && !memoryMetadataClass &&
        !retimeModulesMetadataClass)
      return {};
    auto builder = mlir::ImplicitLocOpBuilder::atBlockEnd(
        mlir::UnknownLoc::get(circtOp->getContext()), circtOp.getBodyBlock());
    SmallVector<PortInfo> mports;
    auto sifiveMetadataClass = builder.create<ClassOp>(
        builder.getStringAttr("SiFive_Metadata"), mports);
    builder.setInsertionPointToStart(sifiveMetadataClass.getBodyBlock());

    auto addPort = [&](Value obj, StringRef fieldName) {
      auto portIndex = sifiveMetadataClass.getNumPorts();
      SmallVector<std::pair<unsigned, PortInfo>> newPorts = {
          {portIndex, PortInfo(builder.getStringAttr(fieldName + "_field_" +
                                                     Twine(portIndex)),
                               obj.getType(), Direction::Out)}};
      sifiveMetadataClass.insertPorts(newPorts);
      auto blockarg = sifiveMetadataClass.getBodyBlock()->addArgument(
          obj.getType(), topMod->getLoc());
      builder.create<PropAssignOp>(blockarg, obj);
    };
    if (blackBoxMetadataClass)
      addPort(
          builder.create<ObjectOp>(blackBoxMetadataClass,
                                   builder.getStringAttr("blackbox_metadata")),
          "blackbox");

    if (memoryMetadataClass)
      addPort(
          builder.create<ObjectOp>(memoryMetadataClass,
                                   builder.getStringAttr("memory_metadata")),
          "memory");

    if (retimeModulesMetadataClass)
      addPort(builder.create<ObjectOp>(
                  retimeModulesMetadataClass,
                  builder.getStringAttr("retime_modules_metadata")),
              "retime");

    if (dutMod) {
      // This can handle multiple DUTs or multiple paths to a DUT.
      // Create a list of paths to the DUTs.
      SmallVector<Value, 2> pathOpsToDut;

      auto dutPaths = instancePathCache.getAbsolutePaths(dutMod);
      // For each path to the DUT.
      for (auto dutPath : dutPaths) {
        SmallVector<Attribute> namepath;
        // Construct the list of inner refs to the instances in the path.
        for (auto inst : dutPath)
          namepath.emplace_back(firrtl::getInnerRefTo(
              inst, [&](auto mod) -> hw::InnerSymbolNamespace & {
                return getModuleNamespace(mod);
              }));
        if (namepath.empty())
          continue;
        // The path op will refer to the leaf instance in the path (and not the
        // actual DUT module!!).
        auto leafInst = dutPath.leaf();
        auto nlaBuilder = OpBuilder::atBlockBegin(circtOp.getBodyBlock());
        auto nla = nlaBuilder.create<hw::HierPathOp>(
            dutMod->getLoc(),
            nlaBuilder.getStringAttr(circtNamespace.newName("dutNLA")),
            nlaBuilder.getArrayAttr(namepath));
        // Create the path ref op and record it.
        pathOpsToDut.emplace_back(createPathRef(leafInst, nla, builder));
      }
      auto *context = builder.getContext();
      // Create the list of paths op and add it as a field of the class.
      auto pathList = builder.create<ListCreateOp>(
          ListType::get(context, cast<PropertyType>(PathType::get(context))),
          pathOpsToDut);
      addPort(pathList, "dutModulePath");
    }

    builder.setInsertionPointToEnd(topMod.getBodyBlock());
    return builder.create<ObjectOp>(sifiveMetadataClass,
                                    builder.getStringAttr("sifive_metadata"));
  }

  /// Get the cached namespace for a module.
  hw::InnerSymbolNamespace &getModuleNamespace(FModuleLike module) {
    return moduleNamespaces.try_emplace(module, module).first->second;
  }
  CircuitOp circtOp;
  FModuleOp dutMod;
  CircuitNamespace circtNamespace;
  InstancePathCache instancePathCache;
  /// Cached module namespaces.
  DenseMap<Operation *, hw::InnerSymbolNamespace> &moduleNamespaces;
  ClassOp memorySchemaClass, extraPortsClass;
  ClassOp memoryMetadataClass;
  ClassOp retimeModulesMetadataClass, retimeModulesSchemaClass;
  ClassOp blackBoxModulesSchemaClass, blackBoxMetadataClass;
  StringRef memoryParamNames[12] = {
      "name",        "depth",      "width",          "maskBits",
      "readPorts",   "writePorts", "readwritePorts", "writeLatency",
      "readLatency", "hierarchy",  "inDut",          "extraPorts"};
  StringRef retimeModulesParamNames[1] = {"moduleName"};
  StringRef blackBoxModulesParamNames[1] = {"moduleName"};
  llvm::SmallDenseSet<StringRef> blackboxModules;
}; // namespace

class CreateSiFiveMetadataPass
    : public circt::firrtl::impl::CreateSiFiveMetadataBase<
          CreateSiFiveMetadataPass> {
  LogicalResult emitRetimeModulesMetadata(ObjectModelIR &omir);
  LogicalResult emitSitestBlackboxMetadata(ObjectModelIR &omir);
  LogicalResult emitMemoryMetadata(ObjectModelIR &omir);
  void runOnOperation() override;

  /// Get the cached namespace for a module.
  hw::InnerSymbolNamespace &getModuleNamespace(FModuleLike module) {
    return moduleNamespaces.try_emplace(module, module).first->second;
  }
  // The set of all modules underneath the design under test module.
  DenseSet<Operation *> dutModuleSet;
  /// Cached module namespaces.
  DenseMap<Operation *, hw::InnerSymbolNamespace> moduleNamespaces;
  // The design under test module.
  FModuleOp dutMod;
  CircuitOp circuitOp;

public:
  CreateSiFiveMetadataPass(bool replSeqMem, StringRef replSeqMemFile) {
    this->replSeqMem = replSeqMem;
    this->replSeqMemFile = replSeqMemFile.str();
  }
};
} // end anonymous namespace

/// This function collects all the firrtl.mem ops and creates a verbatim op with
/// the relevant memory attributes.
LogicalResult
CreateSiFiveMetadataPass::emitMemoryMetadata(ObjectModelIR &omir) {
  if (!replSeqMem)
    return success();

  // Everything goes in the DUT if (1) there is no DUT specified or (2) if the
  // DUT is the top module.
  bool everythingInDUT =
      !dutMod ||
      omir.instancePathCache.instanceGraph.getTopLevelNode()->getModule() ==
          dutMod;
  SmallDenseMap<Attribute, unsigned> symbolIndices;
  auto addSymbolToVerbatimOp =
      [&](Operation *op,
          llvm::SmallVectorImpl<Attribute> &symbols) -> SmallString<8> {
    Attribute symbol;
    if (auto module = dyn_cast<FModuleLike>(op))
      symbol = FlatSymbolRefAttr::get(module);
    else
      symbol = firrtl::getInnerRefTo(
          op, [&](auto mod) -> hw::InnerSymbolNamespace & {
            return getModuleNamespace(mod);
          });

    auto [it, inserted] = symbolIndices.try_emplace(symbol, symbols.size());
    if (inserted)
      symbols.push_back(symbol);

    SmallString<8> str;
    ("{{" + Twine(it->second) + "}}").toVector(str);
    return str;
  };
  // This lambda, writes to the given Json stream all the relevant memory
  // attributes. Also adds the memory attrbutes to the string for creating the
  // memmory conf file.
  auto createMemMetadata = [&](FMemModuleOp mem,
                               llvm::json::OStream &jsonStream,
                               std::string &seqMemConfStr,
                               SmallVectorImpl<Attribute> &jsonSymbols,
                               SmallVectorImpl<Attribute> &seqMemSymbols) {
    bool inDut = everythingInDUT || dutModuleSet.contains(mem);
    omir.addMemory(mem, inDut);
    // Get the memory data width.
    auto width = mem.getDataWidth();
    // Metadata needs to be printed for memories which are candidates for
    // macro replacement. The requirements for macro replacement::
    // 1. read latency and write latency of one.
    // 2. undefined read-under-write behavior.
    if (mem.getReadLatency() != 1 || mem.getWriteLatency() != 1 || width <= 0)
      return;
    auto memExtSym = FlatSymbolRefAttr::get(SymbolTable::getSymbolName(mem));
    auto symId = seqMemSymbols.size();
    seqMemSymbols.push_back(memExtSym);
    // Compute the mask granularity.
    auto isMasked = mem.isMasked();
    auto maskGran = width;
    if (isMasked)
      maskGran /= mem.getMaskBits();
    // Now create the config string for the memory.
    std::string portStr;
    for (uint32_t i = 0; i < mem.getNumWritePorts(); ++i) {
      if (!portStr.empty())
        portStr += ",";
      portStr += isMasked ? "mwrite" : "write";
    }
    for (uint32_t i = 0; i < mem.getNumReadPorts(); ++i) {
      if (!portStr.empty())
        portStr += ",";
      portStr += "read";
    }
    for (uint32_t i = 0; i < mem.getNumReadWritePorts(); ++i) {
      if (!portStr.empty())
        portStr += ",";
      portStr += isMasked ? "mrw" : "rw";
    }

    auto maskGranStr =
        !isMasked ? "" : " mask_gran " + std::to_string(maskGran);
    seqMemConfStr = (StringRef(seqMemConfStr) + "name {{" + Twine(symId) +
                     "}} depth " + Twine(mem.getDepth()) + " width " +
                     Twine(width) + " ports " + portStr + maskGranStr + "\n")
                        .str();

    // Do not emit any JSON for memories which are not in the DUT.
    if (!everythingInDUT && !dutModuleSet.contains(mem))
      return;
    // This adds a Json array element entry corresponding to this memory.
    jsonStream.object([&] {
      jsonStream.attribute("module_name",
                           addSymbolToVerbatimOp(mem, jsonSymbols));
      jsonStream.attribute("depth", (int64_t)mem.getDepth());
      jsonStream.attribute("width", (int64_t)width);
      jsonStream.attribute("masked", isMasked);
      jsonStream.attribute("read", mem.getNumReadPorts());
      jsonStream.attribute("write", mem.getNumWritePorts());
      jsonStream.attribute("readwrite", mem.getNumReadWritePorts());
      if (isMasked)
        jsonStream.attribute("mask_granularity", (int64_t)maskGran);
      jsonStream.attributeArray("extra_ports", [&] {
        for (auto attr : mem.getExtraPorts()) {
          jsonStream.object([&] {
            auto port = cast<DictionaryAttr>(attr);
            auto name = port.getAs<StringAttr>("name").getValue();
            jsonStream.attribute("name", name);
            auto direction = port.getAs<StringAttr>("direction").getValue();
            jsonStream.attribute("direction", direction);
            auto width = port.getAs<IntegerAttr>("width").getUInt();
            jsonStream.attribute("width", width);
          });
        }
      });
      // Record all the hierarchy names.
      SmallVector<std::string> hierNames;
      jsonStream.attributeArray("hierarchy", [&] {
        // Get the absolute path for the parent memory, to create the
        // hierarchy names.
        auto paths = omir.instancePathCache.getAbsolutePaths(mem);
        for (auto p : paths) {
          if (p.empty())
            continue;

          auto top = p.top();
          std::string hierName =
              addSymbolToVerbatimOp(top->getParentOfType<FModuleOp>(),
                                    jsonSymbols)
                  .c_str();
          auto finalInst = p.leaf();
          for (auto inst : llvm::drop_end(p)) {
            auto parentModule = inst->getParentOfType<FModuleOp>();
            if (dutMod == parentModule)
              hierName =
                  addSymbolToVerbatimOp(parentModule, jsonSymbols).c_str();

            hierName = hierName + "." +
                       addSymbolToVerbatimOp(inst, jsonSymbols).c_str();
          }
          hierName += ("." + finalInst.getInstanceName()).str();

          hierNames.push_back(hierName);
          // Only include the memory path if it is under the DUT or we are in a
          // situation where everything is deemed to be "in the DUT", i.e., when
          // the DUT is the top module or when no DUT is specified.
          if (everythingInDUT ||
              llvm::any_of(p, [&](circt::igraph::InstanceOpInterface inst) {
                return llvm::all_of(inst.getReferencedModuleNamesAttr(),
                                    [&](Attribute attr) {
                                      return attr == dutMod.getNameAttr();
                                    });
              }))
            jsonStream.value(hierName);
        }
      });
    });
  };

  std::string dutJsonBuffer;
  llvm::raw_string_ostream dutOs(dutJsonBuffer);
  llvm::json::OStream dutJson(dutOs, 2);
  SmallVector<Attribute, 8> seqMemSymbols;
  SmallVector<Attribute, 8> jsonSymbols;

  std::string seqMemConfStr;
  dutJson.array([&] {
    for (auto mem : circuitOp.getOps<FMemModuleOp>())
      createMemMetadata(mem, dutJson, seqMemConfStr, jsonSymbols,
                        seqMemSymbols);
  });

  auto *context = &getContext();
  auto builder = ImplicitLocOpBuilder::atBlockEnd(UnknownLoc::get(context),
                                                  circuitOp.getBodyBlock());
  AnnotationSet annos(circuitOp);
  auto dirAnno = annos.getAnnotation(metadataDirectoryAttrName);
  StringRef metadataDir = "metadata";
  if (dirAnno)
    if (auto dir = dirAnno.getMember<StringAttr>("dirname"))
      metadataDir = dir.getValue();

  // Use unknown loc to avoid printing the location in the metadata files.
  {
    SmallString<128> seqMemsJsonPath(metadataDir);
    llvm::sys::path::append(seqMemsJsonPath, "seq_mems.json");
    builder.create<emit::FileOp>(seqMemsJsonPath, [&] {
      builder.create<sv::VerbatimOp>(dutJsonBuffer, ValueRange{},
                                     builder.getArrayAttr(jsonSymbols));
    });
  }

  {
    if (replSeqMemFile.empty()) {
      emitError(circuitOp->getLoc())
          << "metadata emission failed, the option "
             "`-repl-seq-mem-file=<filename>` is mandatory for specifying a "
             "valid seq mem metadata file";
      return failure();
    }

    builder.create<emit::FileOp>(replSeqMemFile, [&] {
      builder.create<sv::VerbatimOp>(seqMemConfStr, ValueRange{},
                                     builder.getArrayAttr(seqMemSymbols));
    });
  }

  return success();
}

/// This will search for a target annotation and remove it from the operation.
/// If the annotation has a filename, it will be returned in the output
/// argument.  If the annotation is missing the filename member, or if more than
/// one matching annotation is attached, it will print an error and return
/// failure.
static LogicalResult removeAnnotationWithFilename(Operation *op,
                                                  StringRef annoClass,
                                                  StringRef &filename) {
  filename = "";
  bool error = false;
  AnnotationSet::removeAnnotations(op, [&](Annotation anno) {
    // If there was a previous error or its not a match, continue.
    if (error || !anno.isClass(annoClass))
      return false;

    // If we have already found a matching annotation, error.
    if (!filename.empty()) {
      op->emitError("more than one ") << annoClass << " annotation attached";
      error = true;
      return false;
    }

    // Get the filename from the annotation.
    auto filenameAttr = anno.getMember<StringAttr>("filename");
    if (!filenameAttr) {
      op->emitError(annoClass) << " requires a filename";
      error = true;
      return false;
    }

    // Require a non-empty filename.
    filename = filenameAttr.getValue();
    if (filename.empty()) {
      op->emitError(annoClass) << " requires a non-empty filename";
      error = true;
      return false;
    }

    return true;
  });

  // If there was a problem above, return failure.
  return failure(error);
}

/// This function collects the name of each module annotated and prints them
/// all as a JSON array.
LogicalResult
CreateSiFiveMetadataPass::emitRetimeModulesMetadata(ObjectModelIR &omir) {

  auto *context = &getContext();

  // Get the filename, removing the annotation from the circuit.
  StringRef filename;
  if (failed(removeAnnotationWithFilename(circuitOp, retimeModulesFileAnnoClass,
                                          filename)))
    return failure();

  if (filename.empty())
    return success();

  // Create a string buffer for the json data.
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  llvm::json::OStream j(os, 2);

  // The output is a json array with each element a module name.
  unsigned index = 0;
  SmallVector<Attribute> symbols;
  SmallString<3> placeholder;
  j.array([&] {
    for (auto module : circuitOp.getBodyBlock()->getOps<FModuleLike>()) {
      // The annotation has no supplemental information, just remove it.
      if (!AnnotationSet::removeAnnotations(module, retimeModuleAnnoClass))
        continue;

      // We use symbol substitution to make sure we output the correct thing
      // when the module goes through renaming.
      j.value(("{{" + Twine(index++) + "}}").str());
      symbols.push_back(SymbolRefAttr::get(module.getModuleNameAttr()));
      omir.addRetimeModule(module);
    }
  });

  // Put the retime information in a verbatim operation.
  auto builder = ImplicitLocOpBuilder::atBlockEnd(UnknownLoc::get(context),
                                                  circuitOp.getBodyBlock());
  builder.create<emit::FileOp>(filename, [&] {
    builder.create<sv::VerbatimOp>(builder.getStringAttr(buffer), ValueRange{},
                                   builder.getArrayAttr(symbols));
  });
  return success();
}

/// This function finds all external modules which will need to be generated for
/// the test harness to run.
LogicalResult
CreateSiFiveMetadataPass::emitSitestBlackboxMetadata(ObjectModelIR &omir) {

  // Any extmodule with these annotations should be excluded from the blackbox
  // list.
  std::array<StringRef, 6> blackListedAnnos = {
      blackBoxAnnoClass, blackBoxInlineAnnoClass, blackBoxPathAnnoClass,
      dataTapsBlackboxClass, memTapBlackboxClass};

  auto *context = &getContext();

  // Get the filenames from the annotations.
  StringRef dutFilename, testFilename;
  if (failed(removeAnnotationWithFilename(circuitOp, sitestBlackBoxAnnoClass,
                                          dutFilename)) ||
      failed(removeAnnotationWithFilename(
          circuitOp, sitestTestHarnessBlackBoxAnnoClass, testFilename)))
    return failure();

  // If we don't have either annotation, no need to run this pass.
  if (dutFilename.empty() && testFilename.empty())
    return success();

  // Find all extmodules in the circuit. Check if they are black-listed from
  // being included in the list. If they are not, separate them into two
  // groups depending on if theyre in the DUT or the test harness.
  SmallVector<StringRef> dutModules;
  SmallVector<StringRef> testModules;
  for (auto extModule : circuitOp.getBodyBlock()->getOps<FExtModuleOp>()) {
    // If the module doesn't have a defname, then we can't record it properly.
    // Just skip it.
    if (!extModule.getDefname())
      continue;

    // If its a generated blackbox, skip it.
    AnnotationSet annos(extModule);
    if (llvm::any_of(blackListedAnnos, [&](auto blackListedAnno) {
          return annos.hasAnnotation(blackListedAnno);
        }))
      continue;

    // Record the defname of the module.
    if (!dutMod || dutModuleSet.contains(extModule)) {
      dutModules.push_back(*extModule.getDefname());
    } else {
      testModules.push_back(*extModule.getDefname());
    }
    omir.addBlackBoxModule(extModule);
  }

  // This is a helper to create the verbatim output operation.
  auto createOutput = [&](SmallVectorImpl<StringRef> &names,
                          StringRef filename) {
    if (filename.empty())
      return;

    // Sort and remove duplicates.
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());

    // The output is a json array with each element a module name. The
    // defname of a module can't change so we can output them verbatim.
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    llvm::json::OStream j(os, 2);
    j.array([&] {
      for (auto &name : names)
        j.value(name);
    });

    // Put the information in a verbatim operation.
    auto builder = ImplicitLocOpBuilder::atBlockEnd(UnknownLoc::get(context),
                                                    circuitOp.getBodyBlock());

    builder.create<emit::FileOp>(filename, [&] {
      builder.create<emit::VerbatimOp>(StringAttr::get(context, buffer));
    });
  };

  createOutput(testModules, testFilename);
  createOutput(dutModules, dutFilename);

  return success();
}

void CreateSiFiveMetadataPass::runOnOperation() {

  auto moduleOp = getOperation();
  auto circuits = moduleOp.getOps<CircuitOp>();
  if (circuits.empty())
    return;
  auto cIter = circuits.begin();
  circuitOp = *cIter++;

  assert(cIter == circuits.end() &&
         "cannot handle more than one CircuitOp in a mlir::ModuleOp");

  auto *body = circuitOp.getBodyBlock();
  // Find the device under test and create a set of all modules underneath it.
  auto it = llvm::find_if(*body, [&](Operation &op) -> bool {
    return AnnotationSet(&op).hasAnnotation(dutAnnoClass);
  });
  auto &instanceGraph = getAnalysis<InstanceGraph>();
  if (it != body->end()) {
    dutMod = dyn_cast<FModuleOp>(*it);
    auto *node = instanceGraph.lookup(cast<igraph::ModuleOpInterface>(*it));
    llvm::for_each(llvm::depth_first(node),
                   [&](igraph::InstanceGraphNode *node) {
                     dutModuleSet.insert(node->getModule());
                   });
  }
  ObjectModelIR omir(circuitOp, dutMod, instanceGraph, moduleNamespaces);

  if (failed(emitRetimeModulesMetadata(omir)) ||
      failed(emitSitestBlackboxMetadata(omir)) ||
      failed(emitMemoryMetadata(omir)))
    return signalPassFailure();
  auto *node = instanceGraph.getTopLevelNode();
  if (FModuleOp topMod = dyn_cast<FModuleOp>(*node->getModule()))
    if (auto objectOp = omir.instantiateSifiveMetadata(topMod)) {
      auto portIndex = topMod.getNumPorts();
      SmallVector<std::pair<unsigned, PortInfo>> ports = {
          {portIndex,
           PortInfo(StringAttr::get(objectOp->getContext(), "metadataObj"),
                    AnyRefType::get(objectOp->getContext()), Direction::Out)}};
      topMod.insertPorts(ports);
      auto builderOM = mlir::ImplicitLocOpBuilder::atBlockEnd(
          topMod->getLoc(), topMod.getBodyBlock());
      auto objectCast = builderOM.create<ObjectAnyRefCastOp>(objectOp);
      builderOM.create<PropAssignOp>(topMod.getArgument(portIndex), objectCast);
    }

  // This pass modifies the hierarchy, InstanceGraph is not preserved.

  // Clear pass-global state as required by MLIR pass infrastructure.
  dutMod = {};
  circuitOp = {};
  dutModuleSet.empty();
}

std::unique_ptr<mlir::Pass>
circt::firrtl::createCreateSiFiveMetadataPass(bool replSeqMem,
                                              StringRef replSeqMemFile) {
  return std::make_unique<CreateSiFiveMetadataPass>(replSeqMem, replSeqMemFile);
}
