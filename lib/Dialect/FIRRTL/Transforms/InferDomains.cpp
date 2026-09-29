//===- InferDomains.cpp - Infer and Check FIRRTL Domains ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// InferDomains implements FIRRTL domain inference and checking.  This pass is
// a bottom-up transform acting on modules.  For each moduleOp, we ensure there
// are no domain crossings, and we make explicit the domain associations of
// ports.
//
// This pass does not require that ExpandWhens has run, but it should have run.
// If ExpandWhens has not been run, then duplicate connections will influence
// domain inference and this can result in errors.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/Debug.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Threading.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <optional>

#define DEBUG_TYPE "firrtl-infer-domains"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_INFERDOMAINS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

using hw::InnerRefNamespace;
using hw::InnerSymbolTableCollection;
using llvm::concat;
using mlir::AsmState;
using mlir::InFlightDiagnostic;
using mlir::ReverseIterator;

namespace {
struct VariableTerm;
} // namespace

//====--------------------------------------------------------------------------
// Helpers.
//====--------------------------------------------------------------------------

using DomainValue = mlir::TypedValue<DomainType>;

using PortInsertions = SmallVector<std::pair<unsigned, PortInfo>>;

/// Pairs of module port indices which are known to refer to the same domain
/// value. These are kept as pairs rather than as terms because the terms are
/// local to a ModuleState. The relationships are instantiated on each
/// internal instance when its containing module is processed.
using DomainPortAliases = SmallVector<std::pair<unsigned, unsigned>>;

/// From a domain info attribute, get the row of associated domains for a
/// hardware value at index i.
static auto getPortDomainAssociation(ArrayAttr info, size_t i) {
  if (info.empty())
    return info.getAsRange<IntegerAttr>();
  return cast<ArrayAttr>(info[i]).getAsRange<IntegerAttr>();
}

/// Return true if the value is a port on the module.
static bool isPort(BlockArgument arg) {
  return isa<FModuleOp>(arg.getOwner()->getParentOp());
}

/// Return true if the value is a port on the module.
static bool isPort(Value value) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return false;
  return isPort(arg);
}

/// Returns true if the value is driven by a connect op.
static bool isDriven(DomainValue port) {
  for (auto *user : port.getUsers())
    if (auto connect = dyn_cast<FConnectLike>(user))
      if (connect.getDest() == port)
        return true;
  return false;
}

/// True if a value of the given type could be associated with a domain.
static bool isHardware(Type type) {
  return type_isa<FIRRTLBaseType, RefType>(type);
}

/// True if the given value could be association with a domain.
static bool isHardware(Value value) { return isHardware(value.getType()); }

//====--------------------------------------------------------------------------
// Global State.
//====--------------------------------------------------------------------------

/// Each domain type declared in the circuit is assigned a type-id, based on the
/// order of declaration. Domain associations for hardware values are
/// represented as a list, or row, of domains. The domains in a row are ordered
/// according to their type's id.
namespace {
struct DomainTypeID {
  size_t index;
};
} // namespace

namespace {
enum class DomainProvenanceKind {
  Constraint,
  Association,
  DomainAlias,
  InstanceBinding,
};

struct DomainProvenanceEdge {
  Value lhs;
  Value rhs;
  Operation *op;
  mlir::Location loc;
  DomainProvenanceKind kind;
  size_t domainIndex;
  bool inferred;
  bool summarized;
};

struct DomainAssignment {
  Value value;
  Value domain;
  size_t domainIndex;
  bool inferred;
};

struct IllegalDomainCrossing {
  Operation *owner;
  Operation *op;
  Value lhs;
  Value rhs;
  DomainValue lhsDomain;
  DomainValue rhsDomain;
  DomainValue lhsSource;
  DomainValue rhsSource;
  size_t domainIndex;
};

using ModulePortDomainInferences =
    SmallVector<std::pair<StringAttr, StringAttr>>;
} // namespace

/// Information about the changes made to the interface of a moduleOp, which can
/// be replayed onto an instance.
namespace {
struct ModuleUpdateInfo {
  /// The updated domain information for a moduleOp.
  ArrayAttr portDomainInfo;
  /// The domain ports which have been inserted into a moduleOp.
  PortInsertions portInsertions;
};
} // namespace

namespace {
struct CircuitState {
  CircuitState(CircuitOp circuit, InstanceGraph &instanceGraph,
               InnerRefNamespace &innerRefNamespace, InferDomainsMode mode,
               StringRef reportJson)
      : circuit(circuit), instanceGraph(instanceGraph),
        innerRefNamespace(innerRefNamespace), mode(mode),
        reportJson(reportJson.str()) {
    processCircuit(circuit);
  }

  LogicalResult run();

  ArrayRef<DomainOp> getDomains() const { return domainTable; }
  size_t getNumDomains() const { return domainTable.size(); }
  DomainOp getDomain(DomainTypeID id) const { return domainTable[id.index]; }
  DomainTypeID getDomainTypeID(Type type) { return typeIDTable[type]; }

  void clearDomainProvenance(Operation *moduleOp) {
    moduleDomainProvenance[moduleOp].clear();
  }
  void recordDomainProvenance(Operation *moduleOp, Value lhs, Value rhs,
                              Operation *op, mlir::Location loc,
                              DomainProvenanceKind kind, size_t domainIndex,
                              bool inferred = false, bool summarized = false) {
    moduleDomainProvenance[moduleOp].push_back(
        {lhs, rhs, op, loc, kind, domainIndex, inferred, summarized});
  }
  bool isInferredModulePortAssociation(StringAttr moduleName,
                                       StringAttr portName,
                                       StringAttr domainPortName) const {
    auto entry = modulePortDomainInferences.find(moduleName);
    return entry != modulePortDomainInferences.end() &&
           llvm::is_contained(entry->second,
                              std::make_pair(portName, domainPortName));
  }
  bool hasExplicitDomainAssociation(Operation *moduleOp, Value value,
                                    size_t domainIndex) const;
  void clearDomainAssignments(Operation *moduleOp) {
    moduleDomainAssignments[moduleOp].clear();
  }
  void recordDomainAssignment(Operation *moduleOp, Value value, Value domain,
                              size_t domainIndex, bool inferred);
  void recordIllegalDomainCrossing(IllegalDomainCrossing crossing) {
    if (shouldEmitDomainReport())
      illegalDomainCrossings.push_back(crossing);
  }
  bool shouldEmitDomainReport() const { return !reportJson.empty(); }
  SmallVector<DomainProvenanceEdge>
  findDomainProvenancePath(Value value, DomainValue domain,
                           size_t domainIndex) const;

  void dirty() { asmState = nullptr; }
  AsmState &getAsmState() {
    if (!asmState) {
      asmState = std::make_unique<AsmState>(
          circuit, mlir::OpPrintingFlags().assumeVerified());
    }
    return *asmState;
  }

  size_t getVariableID(VariableTerm *term) {
    return variableIDTable.insert({term, variableIDTable.size() + 1})
        .first->second;
  }

  DenseMap<StringAttr, ModuleUpdateInfo> &getModuleUpdateTable() {
    return moduleUpdateTable;
  }

  DenseMap<StringAttr, DomainPortAliases> &getModuleDomainPortAliases() {
    return moduleDomainPortAliases;
  }

  DenseMap<StringAttr, ModulePortDomainInferences> &
  getModulePortDomainInferences() {
    return modulePortDomainInferences;
  }

  InnerRefNamespace &getInnerRefNamespace() { return innerRefNamespace; }

  DenseSet<Value> inserted;

private:
  LogicalResult runOnModule(Operation *moduleOp);
  LogicalResult materializeOnModule(Operation *moduleOp);
  LogicalResult writeDomainReport(bool complete);

  void processDomain(DomainOp op) {
    auto index = domainTable.size();
    auto domainType = DomainType::getFromDomainOp(op);
    domainTable.push_back(op);
    typeIDTable.insert({domainType, {index}});
  }

  void processCircuit(CircuitOp circuit) {
    for (auto decl : circuit.getOps<DomainOp>())
      processDomain(decl);
  }

  CircuitOp circuit;
  InstanceGraph &instanceGraph;
  InnerRefNamespace &innerRefNamespace;
  InferDomainsMode mode;
  std::string reportJson;
  SmallVector<DomainOp> domainTable;
  DenseMap<Type, DomainTypeID> typeIDTable;
  DenseMap<VariableTerm *, size_t> variableIDTable;
  std::unique_ptr<AsmState> asmState;
  DenseMap<StringAttr, ModuleUpdateInfo> moduleUpdateTable;
  DenseMap<StringAttr, DomainPortAliases> moduleDomainPortAliases;
  DenseMap<StringAttr, ModulePortDomainInferences> modulePortDomainInferences;
  llvm::MapVector<Operation *, SmallVector<DomainProvenanceEdge>>
      moduleDomainProvenance;
  llvm::MapVector<Operation *, SmallVector<DomainAssignment>>
      moduleDomainAssignments;
  SmallVector<IllegalDomainCrossing> illegalDomainCrossings;
};
} // namespace

SmallVector<DomainProvenanceEdge>
CircuitState::findDomainProvenancePath(Value value, DomainValue domain,
                                       size_t domainIndex) const {
  SmallVector<DomainProvenanceEdge> edges;
  for (auto &entry : moduleDomainProvenance)
    llvm::append_range(edges, entry.second);

  DenseMap<Value, SmallVector<unsigned>> adjacency;
  for (auto [index, edge] : llvm::enumerate(edges)) {
    adjacency[edge.lhs].push_back(index);
    adjacency[edge.rhs].push_back(index);
  }

  struct Previous {
    Value value;
    unsigned edgeIndex;
  };
  DenseMap<Value, Previous> previous;
  SmallVector<Value> worklist{value};
  previous.insert({value, Previous{Value(), 0}});
  for (size_t i = 0; i < worklist.size(); ++i) {
    auto current = worklist[i];
    if (current == domain)
      break;
    for (auto edgeIndex : adjacency.lookup(current)) {
      const auto &edge = edges[edgeIndex];
      if (edge.domainIndex != domainIndex ||
          (edge.kind == DomainProvenanceKind::Association && edge.summarized))
        continue;
      auto other = edge.lhs == current ? edge.rhs : edge.lhs;
      if (previous.contains(other))
        continue;
      previous.insert({other, Previous{current, edgeIndex}});
      worklist.push_back(other);
    }
  }

  if (!previous.contains(domain))
    return {};

  SmallVector<DomainProvenanceEdge> path;
  for (Value current = domain; current != value;) {
    auto prev = previous.lookup(current);
    path.push_back(edges[prev.edgeIndex]);
    current = prev.value;
  }
  std::reverse(path.begin(), path.end());
  return path;
}

bool CircuitState::hasExplicitDomainAssociation(Operation *moduleOp,
                                                Value value,
                                                size_t domainIndex) const {
  auto it = moduleDomainProvenance.find(moduleOp);
  if (it == moduleDomainProvenance.end())
    return false;
  return llvm::any_of(it->second, [&](const DomainProvenanceEdge &edge) {
    return edge.kind == DomainProvenanceKind::Association && !edge.inferred &&
           edge.domainIndex == domainIndex &&
           (edge.lhs == value || edge.rhs == value);
  });
}

void CircuitState::recordDomainAssignment(Operation *moduleOp, Value value,
                                          Value domain, size_t domainIndex,
                                          bool inferred) {
  moduleDomainAssignments[moduleOp].push_back(
      {value, domain, domainIndex, inferred});
}

static void collectFileLocations(Location loc, llvm::json::Array &locations) {
  if (auto fileLoc = dyn_cast<mlir::FileLineColLoc>(loc)) {
    locations.push_back(llvm::json::Object{
        {"file", fileLoc.getFilename().getValue().str()},
        {"line", static_cast<int64_t>(fileLoc.getLine())},
        {"column", static_cast<int64_t>(fileLoc.getColumn())}});
    return;
  }
  if (auto fileRange = dyn_cast<mlir::FileLineColRange>(loc)) {
    locations.push_back(llvm::json::Object{
        {"file", fileRange.getFilename().getValue().str()},
        {"start_line", static_cast<int64_t>(fileRange.getStartLine())},
        {"start_column", static_cast<int64_t>(fileRange.getStartColumn())},
        {"end_line", static_cast<int64_t>(fileRange.getEndLine())},
        {"end_column", static_cast<int64_t>(fileRange.getEndColumn())}});
    return;
  }
  if (auto callSiteLoc = dyn_cast<mlir::CallSiteLoc>(loc)) {
    collectFileLocations(callSiteLoc.getCallee(), locations);
    collectFileLocations(callSiteLoc.getCaller(), locations);
    return;
  }
  if (auto nameLoc = dyn_cast<mlir::NameLoc>(loc)) {
    collectFileLocations(nameLoc.getChildLoc(), locations);
    return;
  }
  if (auto fusedLoc = dyn_cast<mlir::FusedLoc>(loc))
    for (auto child : fusedLoc.getLocations())
      collectFileLocations(child, locations);
}

static llvm::json::Object locationToJSON(Location loc) {
  std::string printed;
  llvm::raw_string_ostream stream(printed);
  loc.print(stream);

  llvm::json::Array locations;
  collectFileLocations(loc, locations);
  return llvm::json::Object{{"display", std::move(printed)},
                            {"sources", std::move(locations)}};
}

static std::string typeToString(Type type) {
  std::string printed;
  llvm::raw_string_ostream stream(printed);
  type.print(stream);
  return printed;
}

static Location getValueLocation(Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (auto module = dyn_cast<FModuleLike>(arg.getOwner()->getParentOp()))
      return module.getPortLocation(arg.getArgNumber());
  }
  if (auto result = dyn_cast<OpResult>(value))
    if (auto instance = dyn_cast<FInstanceLike>(result.getOwner()))
      return instance.getPortLocation(result.getResultNumber());
  return value.getLoc();
}

static std::string getValueName(Value value, StringRef fallback) {
  auto name = getFieldName(value).first;
  if (!name.empty())
    return name;
  if (auto arg = dyn_cast<BlockArgument>(value))
    if (auto module = dyn_cast<FModuleLike>(arg.getOwner()->getParentOp()))
      return module.getPortName(arg.getArgNumber()).str();
  if (auto result = dyn_cast<OpResult>(value))
    if (auto instance = dyn_cast<FInstanceLike>(result.getOwner()))
      return instance.getPortName(result.getResultNumber()).str();

  return fallback.str();
}

LogicalResult CircuitState::writeDomainReport(bool complete) {
  using llvm::json::Array;
  using llvm::json::Object;
  using JsonValue = llvm::json::Value;
  using ID = int64_t;

  struct ModuleReport {
    FModuleLike module;
    ID id;
    SmallVector<Value> values;
    SmallVector<FInstanceLike> instances;
  };

  SmallVector<ModuleReport> modules;
  DenseMap<Operation *, ID> moduleIDs;
  DenseMap<Operation *, ID> instanceIDs;
  DenseMap<Value, ID> valueIDs;
  DenseMap<Location, ID> locationIDs;
  DenseMap<Type, ID> typeIDs;
  DenseMap<Operation *, ID> operationKindIDByOp;
  llvm::StringMap<ID> operationKindIDs;
  SmallVector<Location> locations;
  SmallVector<Type> types;
  SmallVector<std::string> operationKindNames;

  for (auto module : circuit.getOps<FModuleLike>()) {
    auto moduleID = static_cast<ID>(modules.size());
    moduleIDs[module.getOperation()] = moduleID;
    modules.push_back({module, moduleID, {}, {}});
  }

  for (auto &moduleReport : modules) {
    auto module = moduleReport.module;
    DenseSet<Value> seenValues;
    auto addValue = [&](Value value) {
      if (!seenValues.insert(value).second)
        return;
      auto valueID = static_cast<ID>(valueIDs.size());
      valueIDs[value] = valueID;
      moduleReport.values.push_back(value);
    };

    if (auto moduleOp = dyn_cast<FModuleOp>(module.getOperation()))
      for (size_t i = 0; i < moduleOp.getNumPorts(); ++i) {
        auto value = moduleOp.getArgument(i);
        addValue(value);
      }

    module->walk([&](Operation *op) {
      if (op != module.getOperation()) {
        if (auto instance = dyn_cast<FInstanceLike>(op)) {
          auto instanceID = static_cast<ID>(instanceIDs.size());
          instanceIDs[op] = instanceID;
          moduleReport.instances.push_back(instance);
        }
        for (auto result : op->getResults())
          if (isHardware(result) || isa<DomainType>(result.getType()))
            addValue(result);
      }
      for (auto &region : op->getRegions())
        for (auto &block : region)
          if (op != module.getOperation())
            for (auto arg : block.getArguments())
              if (isHardware(arg) || isa<DomainType>(arg.getType()))
                addValue(arg);
    });
  }

  auto getLocationID = [&](Location location) -> ID {
    auto it = locationIDs.find(location);
    if (it != locationIDs.end())
      return it->second;
    auto id = static_cast<ID>(locations.size());
    locationIDs[location] = id;
    locations.push_back(location);
    return id;
  };
  auto getTypeID = [&](Type type) -> ID {
    auto it = typeIDs.find(type);
    if (it != typeIDs.end())
      return it->second;
    auto id = static_cast<ID>(types.size());
    typeIDs[type] = id;
    types.push_back(type);
    return id;
  };
  auto getOperationKindID = [&](Operation *op) -> ID {
    auto opIt = operationKindIDByOp.find(op);
    if (opIt != operationKindIDByOp.end())
      return opIt->second;
    auto name = op->getName().getStringRef();
    auto kindIt = operationKindIDs.find(name);
    ID id;
    if (kindIt != operationKindIDs.end()) {
      id = kindIt->second;
    } else {
      id = static_cast<ID>(operationKindIDs.size());
      operationKindIDs[name] = id;
      operationKindNames.push_back(name.str());
    }
    operationKindIDByOp[op] = id;
    return id;
  };
  auto getValueID = [&](mlir::Value value) -> std::optional<ID> {
    auto it = valueIDs.find(value);
    if (it == valueIDs.end())
      return std::nullopt;
    return it->second;
  };
  auto getValueIDJSON = [&](mlir::Value value) -> JsonValue {
    if (auto id = getValueID(value))
      return *id;
    return nullptr;
  };

  Array domainJSON;
  for (auto [index, domain] : llvm::enumerate(domainTable)) {
    domainJSON.push_back(
        Object{{"id", static_cast<int64_t>(index)},
               {"name", domain.getNameAttr().getValue().str()},
               {"location_id", getLocationID(domain.getLoc())}});
  }

  Array moduleJSON;
  for (auto &moduleReport : modules) {
    auto module = moduleReport.module;
    auto moduleName = module.getModuleNameAttr().getValue();
    Array valuesJSON;
    for (auto value : moduleReport.values) {
      auto valueID = valueIDs.lookup(value);
      auto fallbackName = (Twine("value#") + Twine(valueID)).str();
      Object valueObject{
          {"id", valueID},
          {"name", getValueName(value, fallbackName)},
          {"kind", isa<DomainType>(value.getType()) ? "domain" : "hardware"},
          {"type_id", getTypeID(value.getType())},
          {"location_id", getLocationID(getValueLocation(value))}};

      if (isa<DomainType>(value.getType())) {
        auto domainIndex = getDomainTypeID(value.getType()).index;
        valueObject["domain_type_id"] = static_cast<int64_t>(domainIndex);
      }
      if (auto arg = dyn_cast<BlockArgument>(value)) {
        if (auto parentModule =
                dyn_cast<FModuleLike>(arg.getOwner()->getParentOp())) {
          auto index = arg.getArgNumber();
          valueObject["port_index"] = static_cast<int64_t>(index);
          valueObject["port_direction"] =
              direction::toString(parentModule.getPortDirection(index)).str();
        }
      } else if (auto result = dyn_cast<OpResult>(value)) {
        if (auto instance = dyn_cast<FInstanceLike>(result.getOwner())) {
          auto index = result.getResultNumber();
          valueObject["instance_port_index"] = static_cast<int64_t>(index);
          valueObject["port_direction"] =
              direction::toString(instance.getPortDirection(index)).str();
          auto instanceID = instanceIDs.find(instance);
          if (instanceID != instanceIDs.end())
            valueObject["instance_id"] = instanceID->second;
        } else if (auto *op = result.getOwner()) {
          valueObject["definition"] = op->getName().getStringRef().str();
        }
      }

      Array assignmentsJSON;
      auto assignmentsIt = moduleDomainAssignments.find(module.getOperation());
      if (assignmentsIt != moduleDomainAssignments.end())
        for (const auto &assignment : assignmentsIt->second) {
          if (assignment.value != value)
            continue;
          Object assignmentObject{
              {"domain_type_id", static_cast<int64_t>(assignment.domainIndex)},
              {"inferred", assignment.inferred}};
          if (auto assignedDomain = getValueID(assignment.domain))
            assignmentObject["domain_value_id"] = *assignedDomain;
          else
            assignmentObject["domain_value_id"] = nullptr;
          assignmentsJSON.push_back(std::move(assignmentObject));
        }
      if (!assignmentsJSON.empty())
        valueObject["domain_assignments"] = std::move(assignmentsJSON);
      valuesJSON.push_back(std::move(valueObject));
    }

    Array portsJSON;
    auto domainInfo = module.getDomainInfoAttr();
    for (size_t i = 0; i < module.getNumPorts(); ++i) {
      Object portObject{{"index", static_cast<int64_t>(i)}};
      if (auto moduleOp = dyn_cast<FModuleOp>(module.getOperation())) {
        portObject["value_id"] = getValueIDJSON(moduleOp.getArgument(i));
      } else {
        portObject["name"] = module.getPortName(i).str();
        portObject["type_id"] = getTypeID(module.getPortType(i));
        portObject["direction"] =
            direction::toString(module.getPortDirection(i)).str();
        portObject["location_id"] = getLocationID(module.getPortLocation(i));
      }
      Array portAssignmentsJSON;
      if (isHardware(module.getPortType(i)))
        for (auto domainPortIndexAttr :
             getPortDomainAssociation(domainInfo, i)) {
          auto domainPortIndex = domainPortIndexAttr.getUInt();
          if (domainPortIndex >= module.getNumPorts() ||
              !isa<DomainType>(module.getPortType(domainPortIndex)))
            continue;
          auto domainIndex =
              getDomainTypeID(module.getPortType(domainPortIndex)).index;
          bool inferred = false;
          if (isa<FModuleOp>(module.getOperation()))
            inferred = isInferredModulePortAssociation(
                module.getModuleNameAttr(), module.getPortNameAttr(i),
                module.getPortNameAttr(domainPortIndex));
          Object assignment{
              {"domain_type_id", static_cast<int64_t>(domainIndex)},
              {"domain_port_index", static_cast<int64_t>(domainPortIndex)},
              {"domain_port_value_id",
               [&]() -> JsonValue {
                 if (auto moduleOp = dyn_cast<FModuleOp>(module.getOperation()))
                   return getValueIDJSON(moduleOp.getArgument(domainPortIndex));
                 return nullptr;
               }()},
              {"inferred", inferred}};
          portAssignmentsJSON.push_back(std::move(assignment));
        }
      if (!portAssignmentsJSON.empty())
        portObject["domain_assignments"] = std::move(portAssignmentsJSON);
      portsJSON.push_back(std::move(portObject));
    }

    Array instancesJSON;
    auto &irns = getInnerRefNamespace();
    for (auto instance : moduleReport.instances) {
      auto instanceID = instanceIDs.lookup(instance);
      auto targetNames = instance.getReferencedModuleNamesAttr();
      Array targetsJSON;
      Array bindingsJSON;
      for (auto targetNameAttr : targetNames.getAsRange<StringAttr>()) {
        auto targetName = targetNameAttr.getValue();
        auto targetOp = irns.symTable.lookup(targetNameAttr);
        auto targetIDIt = moduleIDs.find(targetOp);
        JsonValue targetID = targetIDIt == moduleIDs.end()
                                 ? JsonValue(targetName.str())
                                 : JsonValue(targetIDIt->second);
        targetsJSON.push_back(std::move(targetID));
        auto target = llvm::dyn_cast_if_present<FModuleLike>(targetOp);
        if (!target || targetIDIt == moduleIDs.end())
          continue;
        auto numPorts = std::min(instance.getNumPorts(), target.getNumPorts());
        for (size_t i = 0; i < numPorts; ++i) {
          if (!isHardware(target.getPortType(i)))
            continue;
          auto instancePort = instance->getResult(i);
          for (auto domainPortIndexAttr :
               getPortDomainAssociation(target.getDomainInfoAttr(), i)) {
            auto domainPortIndex = domainPortIndexAttr.getUInt();
            if (domainPortIndex >= instance.getNumPorts() ||
                !isa<DomainType>(
                    instance->getResult(domainPortIndex).getType()))
              continue;
            auto domainIndex =
                getDomainTypeID(target.getPortType(domainPortIndex)).index;
            auto domainValue = instance->getResult(domainPortIndex);
            Object binding{
                {"target_module_id", targetIDIt->second},
                {"port_index", static_cast<int64_t>(i)},
                {"port_value_id", getValueIDJSON(instancePort)},
                {"domain_type_id", static_cast<int64_t>(domainIndex)},
                {"domain_port_index", static_cast<int64_t>(domainPortIndex)},
                {"effective_domain_value_id", getValueIDJSON(domainValue)},
                {"effective_domain_value_name", getValueName(domainValue, "")},
                {"location_id", getLocationID(instance.getPortLocation(i))}};
            bindingsJSON.push_back(std::move(binding));
          }
        }
      }
      instancesJSON.push_back(
          Object{{"id", instanceID},
                 {"name", instance.getInstanceName().str()},
                 {"targets", std::move(targetsJSON)},
                 {"location_id", getLocationID(instance.getLoc())},
                 {"effective_domain_bindings", std::move(bindingsJSON)}});
    }

    moduleJSON.push_back(
        Object{{"id", moduleReport.id},
               {"name", moduleName.str()},
               {"kind", isa<FExtModuleOp>(module.getOperation()) ? "extmodule"
                                                                 : "module"},
               {"ports", std::move(portsJSON)},
               {"values", std::move(valuesJSON)},
               {"instances", std::move(instancesJSON)}});
  }

  // Keep edge records as positional arrays to avoid repeating object keys.
  Array edgeKindsJSON{"constraint", "association", "domain_alias",
                      "instance_binding"};
  Array edgeFieldsJSON{"owner_module_id", "kind_id",
                       "domain_type_id",  "location_id",
                       "flags",           "lhs_value_id",
                       "rhs_value_id",    "operation_kind_id",
                       "instance_id",     "target_module_id"};

  // Intern source locations and operation kinds before serializing the tables.
  for (auto &entry : moduleDomainProvenance)
    for (const auto &edge : entry.second) {
      getLocationID(edge.loc);
      if (edge.op)
        getOperationKindID(edge.op);
    }
  for (const auto &crossing : illegalDomainCrossings) {
    getLocationID(crossing.op->getLoc());
    getOperationKindID(crossing.op);
  }

  Array typesJSON;
  for (auto type : types)
    typesJSON.push_back(typeToString(type));

  Array locationsJSON;
  for (auto location : locations)
    locationsJSON.push_back(locationToJSON(location));

  Array operationKindsJSON;
  for (const auto &operationKind : operationKindNames)
    operationKindsJSON.push_back(operationKind);

  Array edgesJSON;
  for (auto &entry : moduleDomainProvenance) {
    auto ownerIt = moduleIDs.find(entry.first);
    if (ownerIt == moduleIDs.end())
      continue;
    for (const auto &edge : entry.second) {
      int64_t kindID = 0;
      switch (edge.kind) {
      case DomainProvenanceKind::Constraint:
        kindID = 0;
        break;
      case DomainProvenanceKind::Association:
        kindID = 1;
        break;
      case DomainProvenanceKind::DomainAlias:
        kindID = 2;
        break;
      case DomainProvenanceKind::InstanceBinding:
        kindID = 3;
        break;
      }
      auto lhsID = getValueID(edge.lhs);
      auto rhsID = getValueID(edge.rhs);
      int64_t flags = (edge.inferred ? 1 : 0) | (edge.summarized ? 2 : 0);
      Array edgeJSON{ownerIt->second,
                     kindID,
                     static_cast<int64_t>(edge.domainIndex),
                     getLocationID(edge.loc),
                     flags,
                     lhsID ? JsonValue(*lhsID) : JsonValue(nullptr),
                     rhsID ? JsonValue(*rhsID) : JsonValue(nullptr),
                     edge.op ? JsonValue(operationKindIDByOp.lookup(edge.op))
                             : JsonValue(nullptr),
                     nullptr,
                     nullptr};

      if (edge.op)
        if (auto instance = dyn_cast<FInstanceLike>(edge.op))
          if (auto instanceIt = instanceIDs.find(instance);
              instanceIt != instanceIDs.end())
            edgeJSON[8] = instanceIt->second;

      if (edge.kind == DomainProvenanceKind::InstanceBinding)
        if (auto rhsArg = dyn_cast<BlockArgument>(edge.rhs))
          if (auto targetModule =
                  dyn_cast<FModuleLike>(rhsArg.getOwner()->getParentOp())) {
            auto targetIt = moduleIDs.find(targetModule.getOperation());
            if (targetIt != moduleIDs.end())
              edgeJSON[9] = targetIt->second;
          }

      edgesJSON.push_back(std::move(edgeJSON));
    }
  }

  Array crossingsJSON;
  for (const auto &crossing : illegalDomainCrossings) {
    auto owner = moduleIDs.find(crossing.owner);
    if (owner == moduleIDs.end())
      continue;
    crossingsJSON.push_back(
        Object{{"owner_module_id", owner->second},
               {"domain_type_id", static_cast<ID>(crossing.domainIndex)},
               {"location_id", getLocationID(crossing.op->getLoc())},
               {"operation_kind_id", operationKindIDByOp.lookup(crossing.op)},
               {"lhs_value_id", getValueIDJSON(crossing.lhs)},
               {"rhs_value_id", getValueIDJSON(crossing.rhs)},
               {"lhs_domain_value_id", getValueIDJSON(crossing.lhsDomain)},
               {"rhs_domain_value_id", getValueIDJSON(crossing.rhsDomain)},
               {"lhs_source_value_id", getValueIDJSON(crossing.lhsSource)},
               {"rhs_source_value_id", getValueIDJSON(crossing.rhsSource)}});
  }

  Object report{
      {"format", "circt-domain-inference"},
      {"version", 3},
      {"complete", complete},
      {"types", std::move(typesJSON)},
      {"locations", std::move(locationsJSON)},
      {"operation_kinds", std::move(operationKindsJSON)},
      {"provenance_edge_kinds", std::move(edgeKindsJSON)},
      {"provenance_edge_fields", std::move(edgeFieldsJSON)},
      {"provenance_edge_flags", Object{{"inferred", 1}, {"summarized", 2}}},
      {"instance_binding_direction", "parent_instance_to_module_template"},
      {"domains", std::move(domainJSON)},
      {"modules", std::move(moduleJSON)},
      {"provenance_edges", std::move(edgesJSON)},
      {"illegal_crossings", std::move(crossingsJSON)}};

  std::error_code error;
  llvm::raw_fd_ostream output(reportJson, error, llvm::sys::fs::OF_Text);
  if (error)
    return circuit.emitError() << "could not open domain report '" << reportJson
                               << "': " << error.message();
  llvm::json::OStream json(output, /*IndentSize=*/2);
  json.value(JsonValue(std::move(report)));
  json.flush();
  if (output.has_error()) {
    auto message = output.error().message();
    output.clear_error();
    return circuit.emitError() << "failed to write domain report '"
                               << reportJson << "': " << message;
  }
  return success();
}

//====--------------------------------------------------------------------------
// Terms: Syntax for unifying domain and domain-rows.
//====--------------------------------------------------------------------------

/// The different sorts of terms in the unification engine.
namespace {
enum class TermKind {
  Variable,
  Value,
  Row,
};
} // namespace

/// A term in the unification engine.
namespace {
struct Term {
  constexpr Term(TermKind kind) : kind(kind) {}
  TermKind kind;
};
} // namespace

/// Helper to define a term kind.
namespace {
template <TermKind K>
struct TermBase : Term {
  static bool classof(const Term *term) { return term->kind == K; }
  TermBase() : Term(K) {}
};
} // namespace

/// An unknown value.
namespace {
struct VariableTerm : public TermBase<TermKind::Variable> {
  VariableTerm() : leader(nullptr) {}
  VariableTerm(Term *leader) : leader(leader) {}
  Term *leader;
};
} // namespace

/// A concrete value defined in the IR.
namespace {
struct ValueTerm : public TermBase<TermKind::Value> {
  ValueTerm(DomainValue value) : value(value) {}
  DomainValue value;
};
} // namespace

/// A row of domains.
namespace {
struct RowTerm : public TermBase<TermKind::Row> {
  RowTerm(ArrayRef<Term *> elements) : elements(elements) {}
  ArrayRef<Term *> elements;
};
} // namespace

//====--------------------------------------------------------------------------
// Module processing: solve for the domain associations of hardware.
//====--------------------------------------------------------------------------

/// A map from unsolved variables to a port index, where that port has not yet
/// been created. Eventually we will have an input domain at the port index,
/// which will be the solution to the recorded variable.
using PendingSolutions = DenseMap<VariableTerm *, unsigned>;

/// A map from local domains to an aliasing port index, where that port has not
/// yet been created. Eventually we will be exporting the domain value at the
/// port index.
using PendingExports = llvm::MapVector<DomainValue, unsigned>;

namespace {
struct PendingUpdates {
  PortInsertions insertions;
  PendingSolutions solutions;
  PendingExports exports;
};
} // namespace

/// A map from domain IR values defined internal to the moduleOp, to ports that
/// alias that domain. These ports make the domain useable as associations of
/// ports, and we say these are exporting ports.
using ExportTable = DenseMap<DomainValue, TinyPtrVector<DomainValue>>;

namespace {
class ModuleState {
public:
  explicit ModuleState(CircuitState &globals) : globals(globals) {}

  ArrayRef<DomainOp> getDomains() { return globals.getDomains(); }
  size_t getNumDomains() { return globals.getNumDomains(); }
  DomainOp getDomain(DomainTypeID id) { return globals.getDomain(id); }
  DomainTypeID getDomainTypeID(Type type) {
    return globals.getDomainTypeID(type);
  }
  DomainTypeID getDomainTypeID(FModuleLike module, size_t i) {
    return globals.getDomainTypeID(module.getPortType(i));
  }
  DomainTypeID getDomainTypeID(FInstanceLike op, size_t i) const {
    return globals.getDomainTypeID(op->getResult(i).getType());
  }
  DomainTypeID getDomainTypeID(DomainValue value) const {
    return globals.getDomainTypeID(value.getType());
  }
  auto &getModuleUpdateTable() { return globals.getModuleUpdateTable(); }
  auto &getModuleDomainPortAliases() {
    return globals.getModuleDomainPortAliases();
  }

  mlir::AsmState &getAsmState() { return globals.getAsmState(); }
  void dirty() { globals.dirty(); }

  template <typename T>
  void render(Operation *op, T &out);
  template <typename T>
  void render(Value value, T &out);
  template <typename T>
  void renderLong(Value value, T &out);
  template <typename T>
  void render(Term *term, T &out);
  template <typename T>
  struct Render;
  template <typename T>
  Render<T> render(T &&subject);
  struct RenderLong;
  RenderLong renderLong(Value value);

  Term *find(Term *x);
  LogicalResult unify(Term *lhs, Term *rhs);
  LogicalResult unify(VariableTerm *x, Term *y);
  LogicalResult unify(ValueTerm *xv, Term *y);
  LogicalResult unify(RowTerm *lhsRow, Term *rhs);
  void solve(Term *lhs, Term *rhs);

  [[nodiscard]] RowTerm *allocRow(size_t size);
  [[nodiscard]] RowTerm *allocRow(ArrayRef<Term *> elements);
  [[nodiscard]] VariableTerm *allocVar();
  [[nodiscard]] ValueTerm *allocVal(DomainValue value);
  template <typename T, typename... Args>
  T *alloc(Args &&...args);
  ArrayRef<Term *> allocArray(ArrayRef<Term *> elements);

  DomainValue getOptUnderlyingDomain(DomainValue value);
  Term *getOptTermForDomain(DomainValue value);
  Term *getTermForDomain(DomainValue value);
  void setTermForDomain(DomainValue value, Term *term);

  Term *getOptDomainAssociation(Value value);
  Term *getDomainAssociation(Value value);
  void setDomainAssociation(Value value, Term *term);

  /// True if the value is "colorless": it is only driven by nodes or primops
  /// whose inputs all terminate in constants, and therefore is not tied to any
  /// domain. A colorless value imposes and inherits no domain constraints and
  /// may be freely consumed by a value in any domain. All ports and wires are
  /// treated as colored. Computed structurally over the SSA graph, memoized per
  /// module.
  bool isColorless(Value value);

  void processDomainDefinition(DomainValue domain);
  RowTerm *getDomainAssociationAsRow(Value value);

  void noteLocation(InFlightDiagnostic &diag, Operation *op);
  void noteDomain(InFlightDiagnostic &diag, DomainValue domain);
  DomainValue noteDomainSource(InFlightDiagnostic &diag, DomainValue domain);
  DomainValue noteDomainSource(InFlightDiagnostic &diag, Term *term);
  void noteDomainInferencePath(InFlightDiagnostic &diag, Value value,
                               DomainValue domain, size_t domainIndex);
  void emitDomainCrossingError(Operation *op, Value lhs, Term *lhsTerm,
                               Value rhs, Term *rhsTerm);
  template <typename T>
  void emitDuplicatePortDomainError(T op, size_t i, DomainTypeID domainTypeID,
                                    IntegerAttr domainPortIndexAttr1,
                                    IntegerAttr domainPortIndexAttr2);
  template <typename T>
  void emitDomainPortInferenceError(T op, size_t i);
  template <typename T>
  void emitAmbiguousPortDomainAssociation(
      T op, const llvm::TinyPtrVector<DomainValue> &exports,
      DomainTypeID typeID, size_t i);
  template <typename T>
  void emitMissingPortDomainAssociationError(T op, DomainTypeID typeID,
                                             size_t i);

  LogicalResult unifyAssociations(Operation *op, Value lhs, Value rhs);
  template <typename T>
  LogicalResult unifyAssociations(Operation *op, T &&range);
  LogicalResult unifyAssociations(Operation *op);

  LogicalResult processModulePorts(FModuleOp moduleOp);
  template <typename T>
  LogicalResult processInstancePorts(T op);
  FInstanceLike fixInstancePorts(FInstanceLike op,
                                 const ModuleUpdateInfo &update);
  LogicalResult processOp(FInstanceLike op);
  LogicalResult processInstanceDomainPortAliases(FInstanceLike op);
  LogicalResult processOp(UnsafeDomainCastOp op);
  LogicalResult processOp(DomainDefineOp op);
  LogicalResult processOp(WireOp op);
  LogicalResult processOp(RWProbeOp op);
  LogicalResult processOp(Operation *op);
  void recordProvenance(Value lhs, Value rhs, Operation *op,
                        DomainProvenanceKind kind, size_t domainIndex,
                        bool inferred = false, bool summarized = false);
  void recordProvenance(Value lhs, Value rhs, Operation *op,
                        DomainProvenanceKind kind, size_t domainIndex,
                        mlir::Location loc, bool inferred = false,
                        bool summarized = false);
  void recordDomainDefinition(DomainDefineOp op);
  void recordInstanceBindings(FInstanceLike op);
  void recordVariableAssociations(VariableTerm *var, DomainValue domain,
                                  Operation *op);
  void recordDomainAssignments(FModuleOp moduleOp);
  LogicalResult processModuleBody(FModuleOp moduleOp);
  LogicalResult processModule(FModuleOp moduleOp);
  LogicalResult materializeModule(FModuleOp moduleOp);
  void recordDomainPortAliases(FModuleOp moduleOp);

  ExportTable initializeExportTable(FModuleOp moduleOp);
  void ensureSolved(Namespace &ns, DomainTypeID typeID, size_t ip,
                    LocationAttr loc, VariableTerm *var,
                    PendingUpdates &pending);
  void ensureExported(Namespace &ns, const ExportTable &exports,
                      DomainTypeID typeID, size_t ip, LocationAttr loc,
                      ValueTerm *val, PendingUpdates &pending);
  void getUpdatesForDomainAssociationOfPort(Namespace &ns,
                                            PendingUpdates &pending,
                                            DomainTypeID typeID, size_t ip,
                                            LocationAttr loc, Term *term,
                                            const ExportTable &exports);
  void getUpdatesForDomainAssociationOfPort(Namespace &ns,
                                            const ExportTable &exports,
                                            size_t ip, LocationAttr loc,
                                            RowTerm *row,
                                            PendingUpdates &pending);
  void getUpdatesForModulePorts(FModuleOp moduleOp, const ExportTable &exports,
                                Namespace &ns, PendingUpdates &pending);
  void getUpdatesForModule(FModuleOp moduleOp, const ExportTable &exports,
                           PendingUpdates &pending);
  void applyUpdatesToModule(FModuleOp moduleOp, ExportTable &exports,
                            const PendingUpdates &pending);
  SmallVector<Attribute> copyPortDomainAssociations(FModuleOp moduleOp,
                                                    ArrayAttr moduleDomainInfo,
                                                    size_t portIndex);
  LogicalResult driveModuleOutputDomainPorts(FModuleOp moduleOp);
  LogicalResult updateModuleDomainInfo(FModuleOp moduleOp,
                                       const ExportTable &exportTable,
                                       ArrayAttr &result);
  DomainValue
  solveVarWithAnonDomain(OpBuilder &builder,
                         DenseMap<DomainValue, DomainValue> &domainsInScope,
                         Operation *user, DomainType type, VariableTerm *var);
  DomainValue
  getDomainInScope(OpBuilder &builder,
                   DenseMap<DomainValue, DomainValue> &domainsInScope,
                   DomainValue domain);
  LogicalResult
  updateInstance(DenseMap<DomainValue, DomainValue> &domainsInScope,
                 FInstanceLike op);
  LogicalResult updateWire(DenseMap<DomainValue, DomainValue> &domainsInScope,
                           WireOp wireOp);
  LogicalResult updateModuleBody(FModuleOp moduleOp);
  LogicalResult updateModule(FModuleOp moduleOp);

  LogicalResult checkModulePorts(FModuleLike moduleOp);
  LogicalResult checkModuleDomainPortDrivers(FModuleOp moduleOp);
  LogicalResult checkInstanceDomainPortDrivers(FInstanceLike op);
  LogicalResult checkModuleBody(FModuleOp moduleOp);

  LogicalResult inferModule(FModuleOp moduleOp);
  LogicalResult checkModule(FModuleOp moduleOp);
  LogicalResult checkModule(FExtModuleOp extModuleOp);
  LogicalResult checkAndInferModule(FModuleOp moduleOp);

private:
  CircuitState &globals;
  Operation *provenanceOwner = nullptr;
  DenseMap<Value, Term *> termTable;
  DenseMap<Value, Term *> associationTable;
  /// Memoization for `isColorless`. Absent = not computed; present = result.
  DenseMap<Value, bool> colorlessTable;
  llvm::BumpPtrAllocator allocator;
};
} // namespace

template <typename T>
void ModuleState::render(Operation *op, T &out) {
  op->print(out, getAsmState());
}

template <typename T>
void ModuleState::render(Value value, T &out) {
  if (!value) {
    out << "null";
    return;
  }

  auto [name, _] = getFieldName(value);
  if (name.empty()) {
    llvm::raw_string_ostream os(name);
    value.printAsOperand(os, globals.getAsmState());
  }
  out << name;
}

template <typename T>
void ModuleState::renderLong(Value value, T &out) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (auto moduleOp = llvm::dyn_cast_if_present<FModuleLike>(
            arg.getOwner()->getParentOp())) {
      out << direction::toLongString(
          moduleOp.getPortDirection(arg.getArgNumber()));
      out << " module port ";
    }
  } else if (auto result = dyn_cast<OpResult>(value)) {
    auto *op = result.getOwner();
    if (auto inst = dyn_cast<FInstanceLike>(op)) {
      out << direction::toLongString(
          inst.getPortDirection(result.getResultNumber()));
      out << " instance port ";
    }
  }

  render(value, out);
}

template <typename T>
// NOLINTNEXTLINE(misc-no-recursion)
void ModuleState::render(Term *term, T &out) {
  if (!term) {
    out << "null";
    return;
  }
  term = find(term);
  if (auto *var = dyn_cast<VariableTerm>(term)) {
    out << "?" << globals.getVariableID(var);
    return;
  }
  if (auto *val = dyn_cast<ValueTerm>(term)) {
    auto value = val->value;
    render(value, out);
    return;
  }
  if (auto *row = dyn_cast<RowTerm>(term)) {
    out << "[";
    llvm::interleaveComma(
        llvm::seq(size_t(0), getNumDomains()), out, [&](auto i) {
          render(row->elements[i], out);
          out << " : " << getDomain(DomainTypeID{i}).getSymName();
        });
    out << "]";
    return;
  }
  out << "unknown";
}

template <typename T>
struct ModuleState::Render {
  ModuleState *state;
  T subject;
};

template <typename T>
ModuleState::Render<T> ModuleState::render(T &&subject) {
  return Render<T>{this, std::forward<T>(subject)};
}

template <typename T>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &out,
                                     ModuleState::Render<T> r) {
  r.state->render(r.subject, out);
  return out;
}

struct ModuleState::RenderLong {
  ModuleState *state;
  Value value;
};

ModuleState::RenderLong ModuleState::renderLong(Value value) {
  return RenderLong{this, value};
}

static Diagnostic &operator<<(Diagnostic &diag, ModuleState::RenderLong r) {
  r.state->renderLong(r.value, diag);
  return diag;
}

void ModuleState::recordProvenance(Value lhs, Value rhs, Operation *op,
                                   DomainProvenanceKind kind,
                                   size_t domainIndex, bool inferred,
                                   bool summarized) {
  if (!provenanceOwner || !lhs || !rhs)
    return;
  auto loc = op ? op->getLoc() : lhs.getLoc();
  recordProvenance(lhs, rhs, op, kind, domainIndex, loc, inferred, summarized);
}

void ModuleState::recordProvenance(Value lhs, Value rhs, Operation *op,
                                   DomainProvenanceKind kind,
                                   size_t domainIndex, mlir::Location loc,
                                   bool inferred, bool summarized) {
  if (!provenanceOwner || !lhs || !rhs)
    return;
  globals.recordDomainProvenance(provenanceOwner, lhs, rhs, op, loc, kind,
                                 domainIndex, inferred, summarized);
}

void ModuleState::recordDomainDefinition(DomainDefineOp op) {
  auto dest = op.getDest();
  recordProvenance(dest, op.getSrc(), op, DomainProvenanceKind::DomainAlias,
                   getDomainTypeID(dest).index);
}

void ModuleState::recordInstanceBindings(FInstanceLike op) {
  auto &irns = globals.getInnerRefNamespace();
  auto names = op.getReferencedModuleNamesAttr().getAsRange<StringAttr>();
  for (auto name : names) {
    auto moduleOp = dyn_cast<FModuleOp>(irns.symTable.lookup(name));
    if (!moduleOp)
      continue;

    auto numPorts = std::min(op.getNumPorts(), moduleOp.getNumPorts());
    for (size_t i = 0; i < numPorts; ++i) {
      auto instancePort = op->getResult(i);
      auto modulePort = moduleOp.getArgument(i);
      if (isa<DomainType>(instancePort.getType()) &&
          isa<DomainType>(modulePort.getType())) {
        auto typeID = getDomainTypeID(instancePort.getType());
        recordProvenance(instancePort, modulePort, op,
                         DomainProvenanceKind::InstanceBinding, typeID.index);
        continue;
      }
      if (!isHardware(instancePort) || !isHardware(modulePort))
        continue;
      for (size_t domainIndex = 0; domainIndex < getNumDomains(); ++domainIndex)
        recordProvenance(instancePort, modulePort, op,
                         DomainProvenanceKind::InstanceBinding, domainIndex);
    }
  }
}

void ModuleState::recordVariableAssociations(VariableTerm *var,
                                             DomainValue domain,
                                             Operation *op) {
  auto *root = find(var);
  auto domainIndex = getDomainTypeID(domain).index;
  for (auto [value, term] : associationTable) {
    auto *association = find(term);
    if (auto *row = dyn_cast<RowTerm>(association)) {
      if (domainIndex < row->elements.size() &&
          find(row->elements[domainIndex]) == root)
        recordProvenance(value, domain, op, DomainProvenanceKind::Association,
                         domainIndex, true);
      continue;
    }
    if (association == root)
      recordProvenance(value, domain, op, DomainProvenanceKind::Association,
                       domainIndex, true);
  }
}

void ModuleState::recordDomainAssignments(FModuleOp moduleOp) {
  if (!globals.shouldEmitDomainReport())
    return;

  globals.clearDomainAssignments(moduleOp);
  for (auto [value, term] : associationTable) {
    auto *row = dyn_cast<RowTerm>(find(term));
    if (!row)
      continue;
    for (auto [domainIndex, domainTerm] : llvm::enumerate(row->elements)) {
      Value domain;
      if (auto *domainValue = dyn_cast<ValueTerm>(find(domainTerm)))
        domain = domainValue->value;
      auto inferred =
          !globals.hasExplicitDomainAssociation(moduleOp, value, domainIndex);
      globals.recordDomainAssignment(moduleOp, value, domain, domainIndex,
                                     inferred);
    }
  }
}

void ModuleState::noteDomainInferencePath(InFlightDiagnostic &diag, Value value,
                                          DomainValue domain,
                                          size_t domainIndex) {
  auto path = globals.findDomainProvenancePath(value, domain, domainIndex);
  if (path.empty())
    return;
  if (path.size() == 1 &&
      path.front().kind == DomainProvenanceKind::Association &&
      !path.front().inferred)
    return;

  auto &header = diag.attachNote(value.getLoc());
  header << "domain inference path from " << renderLong(value) << " to "
         << renderLong(domain);
  for (const auto &edge : path) {
    auto &note = diag.attachNote(edge.loc);
    switch (edge.kind) {
    case DomainProvenanceKind::Constraint:
      note << "domains of " << renderLong(edge.lhs) << " and "
           << renderLong(edge.rhs) << " are constrained to match";
      if (edge.op)
        note << " by " << edge.op->getName().getStringRef();
      break;
    case DomainProvenanceKind::Association: {
      auto hardware = isa<DomainType>(edge.lhs.getType()) ? edge.rhs : edge.lhs;
      auto associatedDomain =
          isa<DomainType>(edge.lhs.getType()) ? edge.lhs : edge.rhs;
      note << renderLong(hardware) << " is associated with "
           << renderLong(associatedDomain);
      break;
    }
    case DomainProvenanceKind::DomainAlias:
      note << renderLong(edge.lhs) << " aliases " << renderLong(edge.rhs);
      break;
    case DomainProvenanceKind::InstanceBinding:
      note << renderLong(edge.lhs) << " is bound to " << renderLong(edge.rhs);
      break;
    }
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
Term *ModuleState::find(Term *x) {
  if (!x)
    return nullptr;

  if (auto *var = dyn_cast<VariableTerm>(x)) {
    if (var->leader == nullptr)
      return var;

    auto *leader = find(var->leader);
    if (leader != var->leader)
      var->leader = leader;
    return leader;
  }

  return x;
}

LogicalResult ModuleState::unify(VariableTerm *x, Term *y) {
  assert(!x->leader);
  x->leader = y;
  return success();
}

LogicalResult ModuleState::unify(ValueTerm *xv, Term *y) {
  if (auto *yv = dyn_cast<VariableTerm>(y)) {
    yv->leader = xv;
    return success();
  }

  if (auto *yv = dyn_cast<ValueTerm>(y))
    return success(xv == yv);

  return failure();
}

// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult ModuleState::unify(RowTerm *lhsRow, Term *rhs) {
  if (auto *rhsVar = dyn_cast<VariableTerm>(rhs)) {
    rhsVar->leader = lhsRow;
    return success();
  }
  if (auto *rhsRow = dyn_cast<RowTerm>(rhs)) {
    for (auto [x, y] : llvm::zip_equal(lhsRow->elements, rhsRow->elements))
      if (failed(unify(x, y)))
        return failure();
    return success();
  }
  return failure();
}

// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult ModuleState::unify(Term *lhs, Term *rhs) {
  if (!lhs || !rhs)
    return success();
  lhs = find(lhs);
  rhs = find(rhs);
  if (lhs == rhs)
    return success();

  LLVM_DEBUG(llvm::dbgs().indent(6)
             << "unify " << render(lhs) << " = " << render(rhs) << "\n");

  if (auto *lhsVar = dyn_cast<VariableTerm>(lhs))
    return unify(lhsVar, rhs);
  if (auto *lhsVal = dyn_cast<ValueTerm>(lhs))
    return unify(lhsVal, rhs);
  if (auto *lhsRow = dyn_cast<RowTerm>(lhs))
    return unify(lhsRow, rhs);
  return failure();
}

void ModuleState::solve(Term *lhs, Term *rhs) {
  [[maybe_unused]] auto result = unify(lhs, rhs);
  assert(result.succeeded());
}

RowTerm *ModuleState::allocRow(size_t size) {
  SmallVector<Term *> elements;
  elements.resize(size);
  return allocRow(elements);
}

RowTerm *ModuleState::allocRow(ArrayRef<Term *> elements) {
  auto ds = allocArray(elements);
  return alloc<RowTerm>(ds);
}

VariableTerm *ModuleState::allocVar() { return alloc<VariableTerm>(); }

ValueTerm *ModuleState::allocVal(DomainValue value) {
  return alloc<ValueTerm>(value);
}

template <typename T, typename... Args>
T *ModuleState::alloc(Args &&...args) {
  static_assert(std::is_base_of_v<Term, T>, "T must be a term");
  return new (allocator) T(std::forward<Args>(args)...);
}

ArrayRef<Term *> ModuleState::allocArray(ArrayRef<Term *> elements) {
  auto size = elements.size();
  if (size == 0)
    return {};

  auto *result = allocator.Allocate<Term *>(size);
  llvm::uninitialized_copy(elements, result);
  for (size_t i = 0; i < size; ++i)
    if (!result[i])
      result[i] = alloc<VariableTerm>();

  return ArrayRef(result, size);
}

DomainValue ModuleState::getOptUnderlyingDomain(DomainValue value) {
  auto *term = getOptTermForDomain(value);
  if (auto *val = llvm::dyn_cast_if_present<ValueTerm>(term))
    return val->value;
  return nullptr;
}

Term *ModuleState::getOptTermForDomain(DomainValue value) {
  assert(isa<DomainType>(value.getType()));
  auto it = termTable.find(value);
  if (it == termTable.end())
    return nullptr;
  return find(it->second);
}

Term *ModuleState::getTermForDomain(DomainValue value) {
  assert(isa<DomainType>(value.getType()));
  if (auto *term = getOptTermForDomain(value))
    return term;
  auto *term = allocVar();
  setTermForDomain(value, term);
  return term;
}

void ModuleState::setTermForDomain(DomainValue value, Term *term) {
  assert(term);
  assert(!termTable.contains(value));
  termTable.insert({value, term});
  LLVM_DEBUG(llvm::dbgs().indent(6)
             << "set " << render(value) << " := " << render(term) << "\n");
}

Term *ModuleState::getOptDomainAssociation(Value value) {
  assert(isHardware(value));
  auto it = associationTable.find(value);
  if (it == associationTable.end())
    return nullptr;
  return find(it->second);
}

Term *ModuleState::getDomainAssociation(Value value) {
  auto *term = getOptDomainAssociation(value);
  assert(term);
  return term;
}

void ModuleState::setDomainAssociation(Value value, Term *term) {
  assert(isHardware(value));
  assert(term);
  term = find(term);
  associationTable.insert({value, term});
  LLVM_DEBUG({
    llvm::dbgs().indent(6) << "set domains(" << render(value)
                           << ") := " << render(term) << "\n";
  });
}

bool ModuleState::isColorless(Value value) {
  // Non-hardware values (domains, properties, indices, ...) never participate
  // in coloring, so treat them as colorless: they impose no constraint.
  if (!isHardware(value))
    return true;

  // Consult the memo table.  A value is visited (expanded) at most once.
  if (auto it = colorlessTable.find(value); it != colorlessTable.end())
    return it->second;

  // Classify a single value structurally, without recursing.  A "look-through"
  // value (a node or a pure primop) is colorless iff all of its hardware
  // operands are colorless.  For every look-through op the operands to explore
  // are exactly all of its operands (a node and a forwarding cast have a single
  // input operand; a pure expression is only look-through when all of its
  // operands are hardware), so the caller can iterate the defining op's operand
  // list directly rather than collecting a subset here.  Everything else is
  // either a colorless constant root or a colored leaf.  In particular, all
  // ports (block arguments, instance results) and wires are colored and must be
  // assigned a domain.
  enum class Kind { Colorless, Colored, LookThrough };
  auto classify = [&](Value v) -> Kind {
    if (!isHardware(v))
      return Kind::Colorless;

    auto *op = v.getDefiningOp();
    // Block arguments (ports) have no defining op and are always colored.
    if (!op)
      return Kind::Colored;

    // Constants are the only colorless roots.
    if (op->hasTrait<OpTrait::ConstantLike>())
      return Kind::Colorless;

    // A node forwards its single input.
    if (isa<NodeOp>(op))
      return Kind::LookThrough;

    // An unsafe domain cast with explicit domain operands is an explicit
    // coloring point and is always colored.  A cast with no domain operands is
    // a pure forwarding cast that inherits colorlessness from its input.
    if (auto castOp = dyn_cast<UnsafeDomainCastOp>(op)) {
      if (!castOp.getDomains().empty())
        return Kind::Colored;
      return Kind::LookThrough;
    }

    // Pure, memory-effect-free expression ops (prim ops, muxes, casts,
    // aggregate projections) fan out to their hardware-typed operands.  For
    // the ops that are eligible to propagate colorlessness (arithmetic and
    // bitwise prim ops, muxes, casts, aggregate projections) every SSA operand
    // is hardware-typed; scalar indices and amounts are attributes, not
    // operands.  A non-hardware SSA operand (a property, domain, or other
    // opaque value, e.g. a `verbatim.expr` substitution) therefore only
    // appears on ops that reference external state, which are colored.  An
    // expression with no hardware operands is likewise a non-constant root
    // (e.g. an `xmr.ref`) and is colored.
    if (isExpression(op) && mlir::isMemoryEffectFree(op)) {
      if (op->getNumOperands() == 0)
        return Kind::Colored;
      for (auto operand : op->getOperands())
        if (!isHardware(operand))
          return Kind::Colored;
      return Kind::LookThrough;
    }

    // Everything else (wires, instance results, registers, memories, explicit
    // domain casts, invalid values, probes, ...) is a colored leaf.
    return Kind::Colored;
  };

  // Iterative post-order DFS.  Each frame tracks a look-through value (a value
  // whose colorlessness is not yet known) and requires exploring its operands.
  // Every look-through op explores all of its operands, so the frame only needs
  // the value (whose defining op supplies the operands) and the index of the
  // _next_ operand to visit.  A value's colorlessness is the conjunction of its
  // operands' colorlessness.  As soon as a colored operand is found the frame
  // short-circuits to colored.  Since look-through values (constants, nodes,
  // primops) reference only dominating SSA operands, the explored subgraph is
  // acyclic (combinational loops only close through wires, which are colored
  // leaves).
  //
  // Note: this DFS is _not_ sufficient to determine colorlessness through
  // nodes.  It is assumed that a post-condition of this pass is that all wires
  // are assigned domains.
  struct Frame {
    // The lookthrough value whose colorlessness is being resolved.  Its
    // defining op supplies the operands to explore; every look-through op
    // explores all of its operands.
    Value value;
    // The index of the next operand to explore.
    unsigned index = 0;
  };
  SmallVector<Frame> stack;

  // Push the first value onto the stack (or exit).
  switch (classify(value)) {
  case Kind::Colored:
    return colorlessTable[value] = false;
  case Kind::Colorless:
    return colorlessTable[value] = true;
  case Kind::LookThrough:
    stack.push_back({value});
    break;
  }

  // Run the DFS.
  while (!stack.empty()) {
    auto &frame = stack.back();
    auto *op = frame.value.getDefiningOp();
    bool colored = false, pushed = false;

    while (frame.index < op->getNumOperands()) {
      Value child = op->getOperand(frame.index);

      // If already resolved, short-circuit on a colored operand or advance.
      if (auto it = colorlessTable.find(child); it != colorlessTable.end()) {
        if (!it->second) {
          colored = true;
          break;
        }
        ++frame.index;
        continue;
      }

      // Classify the operand.  Classify if not lookthrough.  Otherwise, push
      // the lookthrough operand onto the stack and break so that we descend
      // into it.
      switch (classify(child)) {
      case Kind::Colored:
        colorlessTable[child] = false;
        colored = true;
        break;
      case Kind::Colorless:
        colorlessTable[child] = true;
        ++frame.index;
        continue;
      case Kind::LookThrough:
        stack.push_back({child});
        pushed = true;
        break;
      }
      break;
    }

    // We hit a lookthrough operand.  Dexcend into this.  We will revisit the
    // current frame.index once we have an answer for that operand.
    if (pushed)
      continue;

    // All operands resolved (or a colored operand short-circuited).  Record the
    // result and pop this frame.
    colorlessTable[frame.value] = !colored;
    stack.pop_back();
  }

  return colorlessTable[value];
}

void ModuleState::processDomainDefinition(DomainValue domain) {
  assert(isa<DomainType>(domain.getType()));
  auto *newTerm = allocVal(domain);
  auto *oldTerm = getOptTermForDomain(domain);
  if (!oldTerm) {
    setTermForDomain(domain, newTerm);
    return;
  }

  [[maybe_unused]] auto result = unify(oldTerm, newTerm);
  assert(result.succeeded());
}

RowTerm *ModuleState::getDomainAssociationAsRow(Value value) {
  assert(isHardware(value));
  auto *term = getOptDomainAssociation(value);

  // If the term is unknown, allocate a fresh row and set the association.
  if (!term) {
    auto *row = allocRow(getNumDomains());
    setDomainAssociation(value, row);
    return row;
  }

  // If the term is already a row, return it.
  if (auto *row = dyn_cast<RowTerm>(term))
    return row;

  // Otherwise, unify the term with a fresh row of domains.
  if (auto *var = dyn_cast<VariableTerm>(term)) {
    auto *row = allocRow(getNumDomains());
    solve(var, row);
    return row;
  }

  assert(false && "unhandled term type");
  return nullptr;
}

void ModuleState::noteLocation(InFlightDiagnostic &diag, Operation *op) {
  auto &note = diag.attachNote(op->getLoc());
  if (auto mod = dyn_cast<FModuleOp>(op)) {
    note << "in module " << mod.getModuleNameAttr();
    return;
  }
  if (auto mod = dyn_cast<FExtModuleOp>(op)) {
    note << "in extmodule " << mod.getModuleNameAttr();
    return;
  }
  if (auto inst = dyn_cast<InstanceOp>(op)) {
    note << "in instance " << inst.getInstanceNameAttr();
    return;
  }
  if (auto inst = dyn_cast<InstanceChoiceOp>(op)) {
    note << "in instance_choice " << inst.getNameAttr();
    return;
  }

  note << "here";
}

void ModuleState::noteDomain(InFlightDiagnostic &diag, DomainValue domain) {
  auto &note = diag.attachNote(domain.getLoc());
  note << renderLong(domain);

  if (globals.inserted.contains(domain)) {
    note << " automatically inserted here";
    return;
  }

  note << " declared here";
}

DomainValue ModuleState::noteDomainSource(InFlightDiagnostic &diag,
                                          DomainValue domain) {
  auto &irns = globals.getInnerRefNamespace();
  SmallVector<FInstanceLike> stack;
  llvm::SmallDenseSet<DomainValue> seen;

  // This is reusing "domain" across iterations of the while loop.

  auto chaseConnect = [&]() {
    for (auto *user : domain.getUsers()) {
      if (auto defineOp = dyn_cast<DomainDefineOp>(user)) {
        if (defineOp.getDest() != domain)
          continue;
        auto src = defineOp.getSrc();
        diag.attachNote(defineOp.getLoc())
            << renderLong(domain) << " aliases " << renderLong(src);
        domain = defineOp.getSrc();
        return true;
      }
    }
    return false;
  };

  auto chaseModulePort = [&]() {
    auto arg = dyn_cast<BlockArgument>(domain);
    if (!arg)
      return false;

    auto module =
        llvm::dyn_cast_if_present<FModuleOp>(arg.getOwner()->getParentOp());
    if (!module)
      return false;

    auto name = module.getModuleNameAttr();
    while (!stack.empty()) {
      auto instance = stack.back();
      stack.pop_back();
      auto referenced = instance.getReferencedModuleNamesAttr().getValue();
      if (llvm::is_contained(referenced, name)) {
        domain = cast<DomainValue>(instance->getResult(arg.getArgNumber()));
        return true;
      }
    }
    return false;
  };

  auto chaseInstancePort = [&]() {
    auto result = dyn_cast<OpResult>(domain);
    if (!result)
      return false;

    auto inst = dyn_cast<FInstanceLike>(result.getOwner());
    if (!inst)
      return false;

    auto index = result.getResultNumber();
    if (inst.getPortDirection(index) == Direction::In)
      return false;

    auto names = inst.getReferencedModuleNamesAttr().getAsRange<StringAttr>();
    for (auto name : names) {
      auto moduleLike = cast<FModuleLike>(irns.symTable.lookup(name));
      if (auto moduleOp = dyn_cast<FModuleOp>(moduleLike.getOperation())) {
        stack.push_back(inst);
        domain = cast<DomainValue>(moduleOp.getArgument(index));
        return true;
      }
    }
    return false;
  };

  auto chaseUnderlying = [&]() {
    if (auto *term = getOptTermForDomain(domain)) {
      if (auto *val = dyn_cast<ValueTerm>(term)) {
        if (domain != val->value) {
          diag.attachNote(domain.getLoc())
              << renderLong(domain) << " aliases " << renderLong(val->value);
          domain = val->value;
          return true;
        }
      }
    }
    return false;
  };

  while (true) {
    auto [it, inserted] = seen.insert(domain);
    if (!inserted)
      return domain;

    noteDomain(diag, domain);
    chaseConnect() || chaseModulePort() || chaseInstancePort() ||
        chaseUnderlying();
  }
}

DomainValue ModuleState::noteDomainSource(InFlightDiagnostic &diag,
                                          Term *term) {
  auto *val = dyn_cast<ValueTerm>(find(term));
  if (!val)
    return nullptr;

  return noteDomainSource(diag, val->value);
}

void ModuleState::emitDomainCrossingError(Operation *op, Value lhs,
                                          Term *lhsTerm, Value rhs,
                                          Term *rhsTerm) {
  auto *lhsRow = cast<RowTerm>(lhsTerm);
  auto *rhsRow = cast<RowTerm>(rhsTerm);
  auto diag =
      op->emitError("illegal domain crossing in operation between operands ");
  render(lhs, diag);
  diag << " and ";
  render(rhs, diag);
  auto &note1 = diag.attachNote(lhs.getLoc());
  render(lhs, note1);
  note1 << " has domains ";
  render(lhsRow, note1);
  auto &note2 = diag.attachNote(rhs.getLoc());
  render(rhs, note2);
  note2 << " has domains ";
  render(rhsRow, note2);

  for (size_t i = 0, e = getNumDomains(); i < e; ++i) {
    auto *lhsDomain = find(lhsRow->elements[i]);
    auto *rhsDomain = find(rhsRow->elements[i]);
    if (lhsDomain == rhsDomain)
      continue;

    if (auto *domain = dyn_cast<ValueTerm>(lhsDomain))
      noteDomainInferencePath(diag, lhs, domain->value, i);
    if (auto *domain = dyn_cast<ValueTerm>(rhsDomain))
      noteDomainInferencePath(diag, rhs, domain->value, i);
    auto lhsSource = noteDomainSource(diag, lhsDomain);
    auto rhsSource = noteDomainSource(diag, rhsDomain);
    auto *lhsValue = dyn_cast<ValueTerm>(lhsDomain);
    auto *rhsValue = dyn_cast<ValueTerm>(rhsDomain);
    if (lhsValue && rhsValue)
      globals.recordIllegalDomainCrossing({provenanceOwner, op, lhs, rhs,
                                           lhsValue->value, rhsValue->value,
                                           lhsSource, rhsSource, i});
  }
}

template <typename T>
void ModuleState::emitDuplicatePortDomainError(
    T op, size_t i, DomainTypeID domainTypeID, IntegerAttr domainPortIndexAttr1,
    IntegerAttr domainPortIndexAttr2) {
  auto portName = op.getPortNameAttr(i);
  auto portLoc = op.getPortLocation(i);
  auto domainDecl = getDomain(domainTypeID);
  auto domainName = domainDecl.getNameAttr();
  auto domainPortIndex1 = domainPortIndexAttr1.getUInt();
  auto domainPortIndex2 = domainPortIndexAttr2.getUInt();
  auto domainPortName1 = op.getPortNameAttr(domainPortIndex1);
  auto domainPortName2 = op.getPortNameAttr(domainPortIndex2);
  auto domainPortLoc1 = op.getPortLocation(domainPortIndex1);
  auto domainPortLoc2 = op.getPortLocation(domainPortIndex2);
  auto diag = emitError(portLoc);
  diag << "duplicate " << domainName << " association for port " << portName;
  auto &note1 = diag.attachNote(domainPortLoc1);
  note1 << "associated with " << domainName << " port " << domainPortName1;
  auto &note2 = diag.attachNote(domainPortLoc2);
  note2 << "associated with " << domainName << " port " << domainPortName2;
  noteLocation(diag, op);
}

/// Emit an error when we fail to infer the concrete domain to drive to a
/// domain port.
template <typename T>
void ModuleState::emitDomainPortInferenceError(T op, size_t i) {
  auto name = op.getPortNameAttr(i);
  auto diag = emitError(op->getLoc());
  auto info = op.getDomainInfo();
  diag << "unable to infer value for undriven domain port " << name;
  for (size_t j = 0, e = op.getNumPorts(); j < e; ++j) {
    if (auto assocs = dyn_cast<ArrayAttr>(info[j])) {
      for (auto assoc : assocs) {
        if (i == cast<IntegerAttr>(assoc).getValue()) {
          auto name = op.getPortNameAttr(j);
          auto loc = op.getPortLocation(j);
          diag.attachNote(loc) << "associated with hardware port " << name;
          break;
        }
      }
    }
  }
  noteLocation(diag, op);
}

template <typename T>
void ModuleState::emitAmbiguousPortDomainAssociation(
    T op, const llvm::TinyPtrVector<DomainValue> &exports, DomainTypeID typeID,
    size_t i) {
  auto portName = op.getPortNameAttr(i);
  auto portLoc = op.getPortLocation(i);
  auto domainDecl = getDomain(typeID);
  auto domainName = domainDecl.getNameAttr();
  auto diag = emitError(portLoc) << "ambiguous " << domainName
                                 << " association for port " << portName;
  for (auto e : exports) {
    auto arg = cast<BlockArgument>(e);
    auto name = op.getPortNameAttr(arg.getArgNumber());
    auto loc = op.getPortLocation(arg.getArgNumber());
    diag.attachNote(loc) << "candidate association " << name;
  }
  noteLocation(diag, op);
}

template <typename T>
void ModuleState::emitMissingPortDomainAssociationError(T op,
                                                        DomainTypeID typeID,
                                                        size_t i) {
  auto domainName = getDomain(typeID).getNameAttr();
  auto portName = op.getPortNameAttr(i);
  auto diag = emitError(op.getPortLocation(i))
              << "missing " << domainName << " association for port "
              << portName;
  noteLocation(diag, op);
}

LogicalResult ModuleState::unifyAssociations(Operation *op, Value lhs,
                                             Value rhs) {
  if (!lhs || !rhs)
    return success();

  if (lhs == rhs)
    return success();

  if (!isHardware(lhs) || !isHardware(rhs))
    return success();

  // Colorless values impose and receive no association: colorless is below
  // every color in the lattice.
  if (isColorless(lhs) || isColorless(rhs))
    return success();

  LLVM_DEBUG({
    llvm::dbgs().indent(6) << "unify domains(" << render(lhs) << ") = domains("
                           << render(rhs) << ")\n";
  });

  auto *lhsTerm = getOptDomainAssociation(lhs);
  auto *rhsTerm = getOptDomainAssociation(rhs);

  if (lhsTerm) {
    if (rhsTerm) {
      if (failed(unify(lhsTerm, rhsTerm))) {
        emitDomainCrossingError(op, lhs, lhsTerm, rhs, rhsTerm);
        return failure();
      }
      for (size_t domainIndex = 0; domainIndex < getNumDomains(); ++domainIndex)
        recordProvenance(lhs, rhs, op, DomainProvenanceKind::Constraint,
                         domainIndex);
      return success();
    }
    setDomainAssociation(rhs, lhsTerm);
    for (size_t domainIndex = 0; domainIndex < getNumDomains(); ++domainIndex)
      recordProvenance(lhs, rhs, op, DomainProvenanceKind::Constraint,
                       domainIndex);
    return success();
  }

  if (rhsTerm) {
    setDomainAssociation(lhs, rhsTerm);
    for (size_t domainIndex = 0; domainIndex < getNumDomains(); ++domainIndex)
      recordProvenance(lhs, rhs, op, DomainProvenanceKind::Constraint,
                       domainIndex);
    return success();
  }

  auto *var = allocVar();
  setDomainAssociation(lhs, var);
  setDomainAssociation(rhs, var);
  for (size_t domainIndex = 0; domainIndex < getNumDomains(); ++domainIndex)
    recordProvenance(lhs, rhs, op, DomainProvenanceKind::Constraint,
                     domainIndex);
  return success();
}

template <typename T>
LogicalResult ModuleState::unifyAssociations(Operation *op, T &&range) {
  Value lhs;
  for (auto rhs : std::forward<T>(range)) {
    if (!isHardware(rhs) || isColorless(rhs))
      continue;
    if (failed(unifyAssociations(op, lhs, rhs)))
      return failure();
    lhs = rhs;
  }

  return success();
}

LogicalResult ModuleState::unifyAssociations(Operation *op) {
  return unifyAssociations(
      op, llvm::concat<Value>(op->getOperands(), op->getResults()));
}

LogicalResult ModuleState::processModulePorts(FModuleOp moduleOp) {
  auto numDomains = getNumDomains();
  auto domainInfo = moduleOp.getDomainInfoAttr();
  auto numPorts = moduleOp.getNumPorts();

  DenseMap<unsigned, DomainTypeID> domainTypeIDTable;
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = dyn_cast<DomainValue>(moduleOp.getArgument(i));
    if (!port)
      continue;

    LLVM_DEBUG(llvm::dbgs().indent(4)
               << "process port " << render(port) << "\n");

    if (moduleOp.getPortDirection(i) == Direction::In)
      processDomainDefinition(port);

    domainTypeIDTable[i] = getDomainTypeID(moduleOp, i);
  }

  for (size_t i = 0; i < numPorts; ++i) {
    BlockArgument port = moduleOp.getArgument(i);
    if (!isHardware(port))
      continue;

    LLVM_DEBUG(llvm::dbgs().indent(4)
               << "process port " << render(port) << "\n");

    SmallVector<IntegerAttr> associations(numDomains);
    for (auto domainPortIndex : getPortDomainAssociation(domainInfo, i)) {
      auto domainTypeID = domainTypeIDTable.at(domainPortIndex.getUInt());
      auto prevDomainPortIndex = associations[domainTypeID.index];
      if (prevDomainPortIndex) {
        emitDuplicatePortDomainError(moduleOp, i, domainTypeID,
                                     prevDomainPortIndex, domainPortIndex);
        return failure();
      }
      associations[domainTypeID.index] = domainPortIndex;
    }

    SmallVector<Term *> elements(numDomains);
    for (size_t domainTypeIndex = 0; domainTypeIndex < numDomains;
         ++domainTypeIndex) {
      auto domainPortIndex = associations[domainTypeIndex];
      if (!domainPortIndex)
        continue;
      auto domainPortValue =
          cast<DomainValue>(moduleOp.getArgument(domainPortIndex.getUInt()));
      elements[domainTypeIndex] = getTermForDomain(domainPortValue);
    }

    auto *domainAssociations = allocRow(elements);
    setDomainAssociation(port, domainAssociations);
    for (size_t domainIndex = 0; domainIndex < numDomains; ++domainIndex) {
      auto domainPortIndex = associations[domainIndex];
      if (!domainPortIndex)
        continue;
      auto domainPortValue =
          cast<DomainValue>(moduleOp.getArgument(domainPortIndex.getUInt()));
      auto inferred = globals.isInferredModulePortAssociation(
          moduleOp.getModuleNameAttr(), moduleOp.getPortNameAttr(i),
          moduleOp.getPortNameAttr(domainPortIndex.getUInt()));
      recordProvenance(port, domainPortValue, moduleOp,
                       DomainProvenanceKind::Association, domainIndex,
                       moduleOp.getPortLocation(i), inferred, inferred);
    }
  }

  return success();
}

template <typename T>
LogicalResult ModuleState::processInstancePorts(T op) {
  auto numDomains = getNumDomains();
  auto domainInfo = op.getDomainInfoAttr();
  auto numPorts = op.getNumPorts();

  DenseMap<unsigned, DomainTypeID> domainTypeIDTable;
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = dyn_cast<DomainValue>(op->getResult(i));
    if (!port)
      continue;

    if (op.getPortDirection(i) == Direction::Out && !getOptTermForDomain(port))
      processDomainDefinition(port);

    domainTypeIDTable[i] = getDomainTypeID(op, i);
  }

  for (size_t i = 0; i < numPorts; ++i) {
    Value port = op->getResult(i);
    if (!isHardware(port))
      continue;

    SmallVector<IntegerAttr> associations(numDomains);
    for (auto domainPortIndex : getPortDomainAssociation(domainInfo, i)) {
      auto domainTypeID = domainTypeIDTable.at(domainPortIndex.getUInt());
      auto prevDomainPortIndex = associations[domainTypeID.index];
      if (prevDomainPortIndex) {
        emitDuplicatePortDomainError(op, i, domainTypeID, prevDomainPortIndex,
                                     domainPortIndex);
        return failure();
      }
      associations[domainTypeID.index] = domainPortIndex;
    }

    SmallVector<Term *> elements(numDomains);
    for (size_t domainTypeIndex = 0; domainTypeIndex < numDomains;
         ++domainTypeIndex) {
      auto domainPortIndex = associations[domainTypeIndex];
      if (!domainPortIndex)
        continue;
      auto domainPortValue =
          cast<DomainValue>(op->getResult(domainPortIndex.getUInt()));
      elements[domainTypeIndex] = getTermForDomain(domainPortValue);
    }

    auto *domainAssociations = allocRow(elements);
    setDomainAssociation(port, domainAssociations);
    for (size_t domainIndex = 0; domainIndex < numDomains; ++domainIndex) {
      auto domainPortIndex = associations[domainIndex];
      if (!domainPortIndex)
        continue;
      auto domainPortValue =
          cast<DomainValue>(op->getResult(domainPortIndex.getUInt()));
      bool inferred = false;
      auto portName = op.getPortNameAttr(i);
      auto domainPortName = op.getPortNameAttr(domainPortIndex.getUInt());
      auto names =
          op.getReferencedModuleNamesAttr().template getAsRange<StringAttr>();
      for (auto name : names) {
        inferred |= globals.isInferredModulePortAssociation(name, portName,
                                                            domainPortName);
      }
      recordProvenance(port, domainPortValue, op,
                       DomainProvenanceKind::Association, domainIndex,
                       op.getPortLocation(i), inferred, inferred);
    }
  }

  return success();
}

LogicalResult ModuleState::processInstanceDomainPortAliases(FInstanceLike op) {
  auto names = op.getReferencedModuleNamesAttr().getAsRange<StringAttr>();
  auto &aliases = getModuleDomainPortAliases();

  DomainPortAliases commonAliases;
  bool first = true;
  for (auto name : names) {
    auto lookup = aliases.find(name);
    // An external module is a hard boundary. For an instance choice, an alias
    // is only sound when every possible target provides it.
    if (lookup == aliases.end()) {
      commonAliases.clear();
      first = false;
      break;
    }

    if (first) {
      commonAliases = lookup->second;
      first = false;
      continue;
    }

    commonAliases.erase(llvm::remove_if(commonAliases,
                                        [&](auto alias) {
                                          return !llvm::is_contained(
                                              lookup->second, alias);
                                        }),
                        commonAliases.end());
  }

  for (auto [lhsIndex, rhsIndex] : commonAliases) {
    auto lhs = dyn_cast<DomainValue>(op->getResult(lhsIndex));
    auto rhs = dyn_cast<DomainValue>(op->getResult(rhsIndex));
    if (!lhs || !rhs)
      continue;

    auto *lhsTerm = getTermForDomain(lhs);
    auto *rhsTerm = getTermForDomain(rhs);
    if (succeeded(unify(lhsTerm, rhsTerm))) {
      recordProvenance(lhs, rhs, op, DomainProvenanceKind::DomainAlias,
                       getDomainTypeID(lhs).index);
      continue;
    }

    auto diag = op->emitOpError()
                << "domain ports " << op.getPortName(lhsIndex) << " and "
                << op.getPortName(rhsIndex) << " must alias";
    noteDomainSource(diag, lhs);
    noteDomainSource(diag, rhs);
    return failure();
  }

  return success();
}

FInstanceLike ModuleState::fixInstancePorts(FInstanceLike op,
                                            const ModuleUpdateInfo &update) {
  auto clone = op.cloneWithInsertedPortsAndReplaceUses(update.portInsertions);
  clone.setDomainInfoAttr(update.portDomainInfo);
  op->erase();
  dirty();
  LLVM_DEBUG(llvm::dbgs().indent(6) << "fixup " << render(clone) << "\n");
  return clone;
}

LogicalResult ModuleState::processOp(FInstanceLike op) {
  auto moduleName =
      cast<StringAttr>(cast<ArrayAttr>(op.getReferencedModuleNamesAttr())[0]);
  auto updateTable = getModuleUpdateTable();
  auto lookup = updateTable.find(moduleName);
  if (lookup != updateTable.end()) {
    auto &update = lookup->second;
    if (op->getNumResults() != update.portDomainInfo.size())
      op = fixInstancePorts(op, update);
    else {
      op.setDomainInfoAttr(update.portDomainInfo);
      dirty();
    }
  }

  recordInstanceBindings(op);
  if (failed(processInstanceDomainPortAliases(op)))
    return failure();

  return processInstancePorts(op);
}

LogicalResult ModuleState::processOp(UnsafeDomainCastOp op) {
  auto domains = op.getDomains();
  if (domains.empty())
    return unifyAssociations(op, op.getInput(), op.getResult());

  auto input = op.getInput();

  SmallVector<Term *> elements(getNumDomains());
  bool hasInputAssociation = isHardware(input) && !isColorless(input);
  if (hasInputAssociation) {
    auto *inputRow = getDomainAssociationAsRow(input);
    elements.assign(inputRow->elements);
  }

  DenseSet<size_t> explicitDomains;
  for (auto value : op.getDomains()) {
    auto domain = cast<DomainValue>(value);
    auto typeID = getDomainTypeID(domain);
    elements[typeID.index] = getTermForDomain(domain);
    explicitDomains.insert(typeID.index);
    recordProvenance(op.getResult(), domain, op,
                     DomainProvenanceKind::Association, typeID.index);
  }

  auto *row = allocRow(elements);
  setDomainAssociation(op.getResult(), row);
  if (hasInputAssociation)
    for (size_t domainIndex = 0; domainIndex < getNumDomains(); ++domainIndex)
      if (!explicitDomains.contains(domainIndex))
        recordProvenance(input, op.getResult(), op,
                         DomainProvenanceKind::Constraint, domainIndex);
  return success();
}

LogicalResult ModuleState::processOp(DomainDefineOp op) {
  auto src = op.getSrc();
  auto dst = op.getDest();

  auto *srcTerm = getTermForDomain(src);
  auto *dstTerm = getTermForDomain(dst);
  if (succeeded(unify(dstTerm, srcTerm))) {
    recordProvenance(dst, src, op, DomainProvenanceKind::DomainAlias,
                     getDomainTypeID(dst).index);
    return success();
  }

  auto diag =
      op->emitOpError()
      << "defines a domain value that was inferred to be a different domain '";
  render(dstTerm, diag);
  diag << "'";

  return failure();
}

LogicalResult ModuleState::processOp(WireOp op) {
  // If the wire has explicit domain operands, seed the domain table with them
  // as constraints. When this op is visited, connections have not yet been
  // processed (wire declarations precede their uses), so the existing row
  // contains only fresh variables that unify unconditionally. Any conflict
  // between an explicit wire domain and a connection's inferred domain is
  // caught later by the connection's own processOp.
  if (op.getDomains().empty())
    return unifyAssociations(op, op.getResults());

  // Build a row with the explicitly-specified domain slots filled in and set
  // it as the association for this wire result.
  SmallVector<Term *> elements(getNumDomains());
  SmallVector<std::pair<DomainValue, size_t>> explicitDomains;
  for (auto domain : op.getDomains()) {
    auto domainValue = cast<DomainValue>(domain);
    auto typeID = getDomainTypeID(domainValue);
    elements[typeID.index] = getTermForDomain(domainValue);
    explicitDomains.push_back({domainValue, typeID.index});
  }

  auto *row = allocRow(elements);
  for (auto result : op.getResults())
    setDomainAssociation(result, row);
  for (auto result : op.getResults())
    for (auto [domain, domainIndex] : explicitDomains)
      recordProvenance(result, domain, op, DomainProvenanceKind::Association,
                       domainIndex);

  return success();
}

LogicalResult ModuleState::processOp(RWProbeOp op) {
  auto target = globals.getInnerRefNamespace().lookup(op.getTarget());

  if (target.isPort()) {
    auto targetOp = cast<FModuleOp>(target.getOp());
    auto targetValue = targetOp.getArgument(target.getPort());
    return unifyAssociations(op, targetValue, op.getResult());
  }

  auto targetOp = cast<hw::InnerSymbolOpInterface>(target.getOp());
  auto targetValue = targetOp.getTargetResult();
  return unifyAssociations(op, targetValue, op.getResult());
}

LogicalResult ModuleState::processOp(Operation *op) {
  LLVM_DEBUG(llvm::dbgs().indent(4) << "process " << render(op) << "\n");
  if (auto instance = dyn_cast<FInstanceLike>(op))
    return processOp(instance);
  if (auto wireOp = dyn_cast<WireOp>(op))
    return processOp(wireOp);
  if (auto cast = dyn_cast<UnsafeDomainCastOp>(op))
    return processOp(cast);
  if (auto def = dyn_cast<DomainDefineOp>(op))
    return processOp(def);
  if (auto probe = dyn_cast<RWProbeOp>(op))
    return processOp(probe);
  if (auto create = dyn_cast<DomainCreateOp>(op)) {
    processDomainDefinition(create);
    return success();
  }
  if (auto createAnon = dyn_cast<DomainCreateAnonOp>(op)) {
    processDomainDefinition(createAnon);
    return success();
  }

  return unifyAssociations(op);
}

LogicalResult ModuleState::processModuleBody(FModuleOp moduleOp) {
  return failure(
      moduleOp.getBody()
          .walk([&](Operation *op) -> WalkResult { return processOp(op); })
          .wasInterrupted());
}

LogicalResult ModuleState::processModule(FModuleOp moduleOp) {
  LLVM_DEBUG(llvm::dbgs().indent(2) << "processing:\n");
  globals.clearDomainProvenance(moduleOp);
  provenanceOwner = moduleOp;
  if (failed(processModulePorts(moduleOp))) {
    recordDomainAssignments(moduleOp);
    return failure();
  }
  if (failed(processModuleBody(moduleOp))) {
    recordDomainAssignments(moduleOp);
    return failure();
  }
  return success();
}

void ModuleState::recordDomainPortAliases(FModuleOp moduleOp) {
  DomainPortAliases aliases;

  for (size_t i = 0, e = moduleOp.getNumPorts(); i < e; ++i) {
    if (!isa<DomainType>(moduleOp.getPortType(i)))
      continue;

    auto lhs = cast<DomainValue>(moduleOp.getArgument(i));
    auto *lhsTerm = find(getTermForDomain(lhs));
    for (size_t j = 0; j < i; ++j) {
      if (moduleOp.getPortType(i) != moduleOp.getPortType(j))
        continue;

      auto rhs = cast<DomainValue>(moduleOp.getArgument(j));
      if (lhsTerm == find(getTermForDomain(rhs))) {
        aliases.push_back({static_cast<unsigned>(j), static_cast<unsigned>(i)});
        recordProvenance(lhs, rhs, moduleOp, DomainProvenanceKind::DomainAlias,
                         getDomainTypeID(lhs).index,
                         moduleOp.getPortLocation(i));
      }
    }
  }

  getModuleDomainPortAliases()[moduleOp.getModuleNameAttr()] =
      std::move(aliases);
}

ExportTable ModuleState::initializeExportTable(FModuleOp moduleOp) {
  ExportTable exports;
  size_t numPorts = moduleOp.getNumPorts();
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = dyn_cast<DomainValue>(moduleOp.getArgument(i));
    if (!port)
      continue;
    auto value = getOptUnderlyingDomain(port);
    if (value)
      exports[value].push_back(port);
  }

  LLVM_DEBUG({
    llvm::dbgs().indent(2) << "domain exports:\n";
    for (auto entry : exports) {
      llvm::dbgs().indent(4) << render(entry.first) << " exported as ";
      llvm::interleaveComma(entry.second, llvm::dbgs(),
                            [&](auto e) { llvm::dbgs() << render(e); });
      llvm::dbgs() << "\n";
    }
  });

  return exports;
}

void ModuleState::ensureSolved(Namespace &ns, DomainTypeID typeID, size_t ip,
                               LocationAttr loc, VariableTerm *var,
                               PendingUpdates &pending) {
  if (pending.solutions.contains(var))
    return;

  auto *context = loc.getContext();
  auto domainDecl = getDomain(typeID);
  auto domainName = domainDecl.getNameAttr();

  auto portName = StringAttr::get(context, ns.newName(domainName.getValue()));
  auto portType = DomainType::getFromDomainOp(domainDecl);
  auto portDirection = Direction::In;
  auto portSym = StringAttr();
  auto portLoc = loc;
  auto portAnnos = std::nullopt;
  // Domain type ports have no associations (domain info is in the type).
  auto portDomainInfo = ArrayAttr::get(context, {});
  PortInfo portInfo(portName, portType, portDirection, portSym, portLoc,
                    portAnnos, portDomainInfo);

  pending.solutions[var] = pending.insertions.size() + ip;
  pending.insertions.push_back({ip, portInfo});
}

void ModuleState::ensureExported(Namespace &ns, const ExportTable &exports,
                                 DomainTypeID typeID, size_t ip,
                                 LocationAttr loc, ValueTerm *val,
                                 PendingUpdates &pending) {
  auto value = val->value;
  assert(isa<DomainType>(value.getType()));
  if (isPort(value) || exports.contains(value) ||
      pending.exports.contains(value))
    return;

  auto *context = loc.getContext();

  auto domainDecl = getDomain(typeID);
  auto domainName = domainDecl.getNameAttr();

  auto portName = StringAttr::get(context, ns.newName(domainName.getValue()));
  auto portType = DomainType::getFromDomainOp(domainDecl);
  auto portDirection = Direction::Out;
  auto portSym = StringAttr();
  auto portAnnos = std::nullopt;
  // Domain type ports have no associations (domain info is in the type).
  auto portDomainInfo = ArrayAttr::get(context, {});
  PortInfo portInfo(portName, portType, portDirection, portSym, loc, portAnnos,
                    portDomainInfo);
  pending.exports[value] = pending.insertions.size() + ip;
  pending.insertions.push_back({ip, portInfo});
}

void ModuleState::getUpdatesForDomainAssociationOfPort(
    Namespace &ns, PendingUpdates &pending, DomainTypeID typeID, size_t ip,
    LocationAttr loc, Term *term, const ExportTable &exports) {
  if (auto *var = dyn_cast<VariableTerm>(term)) {
    ensureSolved(ns, typeID, ip, loc, var, pending);
    return;
  }
  if (auto *val = dyn_cast<ValueTerm>(term)) {
    ensureExported(ns, exports, typeID, ip, loc, val, pending);
    return;
  }
  llvm_unreachable("invalid domain association");
}

void ModuleState::getUpdatesForDomainAssociationOfPort(
    Namespace &ns, const ExportTable &exports, size_t ip, LocationAttr loc,
    RowTerm *row, PendingUpdates &pending) {
  for (auto [index, term] : llvm::enumerate(row->elements))
    getUpdatesForDomainAssociationOfPort(ns, pending, DomainTypeID{index}, ip,
                                         loc, find(term), exports);
}

void ModuleState::getUpdatesForModulePorts(FModuleOp moduleOp,
                                           const ExportTable &exports,
                                           Namespace &ns,
                                           PendingUpdates &pending) {
  for (size_t i = 0, e = moduleOp.getNumPorts(); i < e; ++i) {
    auto port = moduleOp.getArgument(i);
    if (!isHardware(port))
      continue;

    getUpdatesForDomainAssociationOfPort(
        ns, exports, i, moduleOp.getPortLocation(i),
        getDomainAssociationAsRow(port), pending);
  }
}

void ModuleState::getUpdatesForModule(FModuleOp moduleOp,
                                      const ExportTable &exports,
                                      PendingUpdates &pending) {
  Namespace ns;
  auto names = moduleOp.getPortNamesAttr();
  for (auto name : names.getAsRange<StringAttr>())
    ns.add(name);
  getUpdatesForModulePorts(moduleOp, exports, ns, pending);
}

void ModuleState::applyUpdatesToModule(FModuleOp moduleOp, ExportTable &exports,
                                       const PendingUpdates &pending) {
  LLVM_DEBUG(llvm::dbgs().indent(2) << "applying updates:\n");
  // Put the domain ports in place.
  moduleOp.insertPorts(pending.insertions);
  dirty();

  // Solve any variables and record them as "self-exporting".
  for (auto [var, portIndex] : pending.solutions) {
    auto portValue = cast<DomainValue>(moduleOp.getArgument(portIndex));
    auto *solution = allocVal(portValue);
    LLVM_DEBUG(llvm::dbgs().indent(4)
               << "new-input " << render(portValue) << "\n");
    recordVariableAssociations(var, portValue, moduleOp);
    solve(var, solution);
    setTermForDomain(portValue, solution);
    exports[portValue].push_back(portValue);
    globals.inserted.insert(portValue);
  }

  // Drive the output ports and record the exports. These definitions establish
  // the terms of newly inserted ports for any later analysis visit. Other
  // generated operations are materialized after the circuit-level analysis
  // has reached a fixed point.
  auto builder = OpBuilder::atBlockEnd(moduleOp.getBodyBlock());
  for (auto [domainValue, portIndex] : pending.exports) {
    auto portValue = cast<DomainValue>(moduleOp.getArgument(portIndex));
    builder.setInsertionPointAfterValue(domainValue);
    auto defineOp = DomainDefineOp::create(builder, portValue.getLoc(),
                                           portValue, domainValue);
    recordDomainDefinition(defineOp);
    exports[domainValue].push_back(portValue);
    globals.inserted.insert(portValue);
    setTermForDomain(portValue, allocVal(domainValue));
    LLVM_DEBUG(llvm::dbgs().indent(4) << "new-output " << render(portValue)
                                      << " := " << render(domainValue) << "\n");
  }
}

SmallVector<Attribute> ModuleState::copyPortDomainAssociations(
    FModuleOp moduleOp, ArrayAttr moduleDomainInfo, size_t portIndex) {
  SmallVector<Attribute> result(getNumDomains());
  auto oldAssociations = getPortDomainAssociation(moduleDomainInfo, portIndex);
  for (auto domainPortIndexAttr : oldAssociations) {
    auto domainPortIndex = domainPortIndexAttr.getUInt();
    auto domainTypeID = getDomainTypeID(moduleOp, domainPortIndex);
    result[domainTypeID.index] = domainPortIndexAttr;
  };
  return result;
}

LogicalResult ModuleState::driveModuleOutputDomainPorts(FModuleOp moduleOp) {
  auto builder = OpBuilder::atBlockEnd(moduleOp.getBodyBlock());
  for (size_t i = 0, e = moduleOp.getNumPorts(); i < e; ++i) {
    auto port = dyn_cast<DomainValue>(moduleOp.getArgument(i));
    if (!port || moduleOp.getPortDirection(i) == Direction::In ||
        isDriven(port))
      continue;

    auto *term = getOptTermForDomain(port);
    auto *val = llvm::dyn_cast_if_present<ValueTerm>(term);
    if (!val) {
      emitDomainPortInferenceError(moduleOp, i);
      return failure();
    }

    auto loc = port.getLoc();
    auto value = val->value;
    LLVM_DEBUG(llvm::dbgs().indent(4) << "connect " << render(port)
                                      << " := " << render(value) << "\n");
    auto defineOp = DomainDefineOp::create(builder, loc, port, value);
    recordDomainDefinition(defineOp);
  }

  return success();
}

LogicalResult ModuleState::updateModuleDomainInfo(
    FModuleOp moduleOp, const ExportTable &exportTable, ArrayAttr &result) {
  // At this point, all domain variables mentioned in ports have been
  // solved by generalizing the moduleOp (adding input domain ports). Now, we
  // have to form the new port domain information for the moduleOp by examining
  // the the associated domains of each port.
  auto *context = moduleOp.getContext();
  auto numDomains = getNumDomains();
  auto oldModuleDomainInfo = moduleOp.getDomainInfoAttr();
  auto numPorts = moduleOp.getNumPorts();
  SmallVector<Attribute> newModuleDomainInfo(numPorts);
  auto &inferredAssociations =
      globals.getModulePortDomainInferences()[moduleOp.getModuleNameAttr()];

  for (size_t i = 0; i < numPorts; ++i) {
    auto port = moduleOp.getArgument(i);
    auto type = port.getType();

    if (isa<DomainType>(type)) {
      // Domain type ports have no associations (domain info is in the type).
      newModuleDomainInfo[i] = ArrayAttr::get(context, {});
      continue;
    }

    if (!isHardware(port)) {
      newModuleDomainInfo[i] = ArrayAttr::get(context, {});
      continue;
    }

    auto associations =
        copyPortDomainAssociations(moduleOp, oldModuleDomainInfo, i);
    auto *row = cast<RowTerm>(getDomainAssociation(port));
    for (size_t domainIndex = 0; domainIndex < numDomains; ++domainIndex) {
      auto domainTypeID = DomainTypeID{domainIndex};
      if (associations[domainIndex])
        continue;

      auto domain = cast<ValueTerm>(find(row->elements[domainIndex]))->value;
      auto &exports = exportTable.at(domain);
      if (exports.empty()) {
        auto portName = moduleOp.getPortNameAttr(i);
        auto portLoc = moduleOp.getPortLocation(i);
        auto domainDecl = getDomain(domainTypeID);
        auto domainName = domainDecl.getNameAttr();
        auto diag = emitError(portLoc) << "private " << domainName
                                       << " association for port " << portName;
        diag.attachNote(domain.getLoc()) << "associated domain: " << domain;
        noteLocation(diag, moduleOp);
        return failure();
      }

      if (exports.size() > 1) {
        emitAmbiguousPortDomainAssociation(moduleOp, exports, domainTypeID, i);
        return failure();
      }

      auto argument = cast<BlockArgument>(exports[0]);
      auto domainPortIndex = argument.getArgNumber();
      associations[domainTypeID.index] =
          IntegerAttr::get(IntegerType::get(context, 32, IntegerType::Unsigned),
                           domainPortIndex);
      auto domainPort =
          cast<DomainValue>(moduleOp.getArgument(domainPortIndex));
      auto association =
          std::make_pair(moduleOp.getPortNameAttr(i),
                         moduleOp.getPortNameAttr(domainPortIndex));
      if (!llvm::is_contained(inferredAssociations, association))
        inferredAssociations.push_back(association);
      recordProvenance(port, domainPort, moduleOp,
                       DomainProvenanceKind::Association, domainIndex,
                       moduleOp.getPortLocation(i), true, true);
    }

    newModuleDomainInfo[i] = ArrayAttr::get(context, associations);
  }

  result = ArrayAttr::get(moduleOp.getContext(), newModuleDomainInfo);
  moduleOp.setDomainInfoAttr(result);
  return success();
}

DomainValue ModuleState::solveVarWithAnonDomain(
    OpBuilder &builder, DenseMap<DomainValue, DomainValue> &domainsInScope,
    Operation *user, DomainType type, VariableTerm *var) {
  auto name = type.getName().getAttr();
  DomainValue anon =
      DomainCreateAnonOp::create(builder, user->getLoc(), type, name);
  dirty();
  LLVM_DEBUG(llvm::dbgs().indent(6) << "create anon " << render(anon) << "\n");
  recordVariableAssociations(var, anon, user);
  solve(var, allocVal(anon));
  domainsInScope[anon] = anon;
  globals.inserted.insert(anon);
  return anon;
}

DomainValue ModuleState::getDomainInScope(
    OpBuilder &builder, DenseMap<DomainValue, DomainValue> &domainsInScope,
    DomainValue domain) {
  auto &domainInScope = domainsInScope[domain];
  if (domainInScope)
    return domainInScope;

  domainInScope = cast<DomainValue>(
      WireOp::create(builder, domain.getLoc(), domain.getType(),
                     domain.getType().getName().getAttr())
          .getResult());

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(domain);
  auto defineOp =
      DomainDefineOp::create(builder, domain.getLoc(), domainInScope, domain);
  recordDomainDefinition(defineOp);
  dirty();
  LLVM_DEBUG(llvm::dbgs().indent(6) << "bounce wire " << render(domainInScope)
                                    << " := " << render(domain) << "\n");
  return domainInScope;
}

LogicalResult
ModuleState::updateInstance(DenseMap<DomainValue, DomainValue> &domainsInScope,
                            FInstanceLike op) {
  LLVM_DEBUG(llvm::dbgs().indent(4) << "update " << render(op) << "\n");
  OpBuilder builder(op.getContext());
  builder.setInsertionPointAfter(op);
  auto numPorts = op->getNumResults();

  for (size_t i = 0; i < numPorts; ++i)
    if (auto port = dyn_cast<DomainValue>(op->getResult(i)))
      if (op.getPortDirection(i) == Direction::Out)
        domainsInScope[port] = port;

  for (size_t i = 0; i < numPorts; ++i) {
    auto port = dyn_cast<DomainValue>(op->getResult(i));
    auto direction = op.getPortDirection(i);
    // If the port is an input domain, we may need to drive the input with
    // a value. If we don't know what value to drive to the port, drive an
    // anonymous domain.
    if (port && direction == Direction::In && !isDriven(port)) {
      auto loc = port.getLoc();
      auto *term = getTermForDomain(port);
      if (auto *var = dyn_cast<VariableTerm>(term)) {
        auto domain = solveVarWithAnonDomain(builder, domainsInScope, op,
                                             port.getType(), var);
        LLVM_DEBUG(llvm::dbgs().indent(6) << "connect " << render(port)
                                          << " := " << render(domain) << "\n");
        auto defineOp = DomainDefineOp::create(builder, loc, port, domain);
        recordDomainDefinition(defineOp);
        continue;
      }
      if (auto *val = dyn_cast<ValueTerm>(term)) {
        auto domain = getDomainInScope(builder, domainsInScope, val->value);
        LLVM_DEBUG(llvm::dbgs().indent(6) << "connect " << render(port)
                                          << " := " << render(domain) << "\n");
        auto defineOp = DomainDefineOp::create(builder, loc, port, domain);
        recordDomainDefinition(defineOp);
        continue;
      }
      llvm_unreachable("unhandled domain term type");
    }
  }

  return success();
}

LogicalResult
ModuleState::updateWire(DenseMap<DomainValue, DomainValue> &domainsInScope,
                        WireOp wireOp) {
  auto result = wireOp.getResult();

  if (auto tgt = dyn_cast<DomainValue>(result)) {
    if (isDriven(tgt))
      return success();

    LLVM_DEBUG(llvm::dbgs().indent(4) << "update " << render(wireOp) << "\n");
    OpBuilder builder(wireOp);
    builder.setInsertionPointAfter(wireOp);
    auto *term = getTermForDomain(tgt);
    if (auto *var = dyn_cast<VariableTerm>(term)) {
      auto src = solveVarWithAnonDomain(builder, domainsInScope, wireOp,
                                        tgt.getType(), var);
      LLVM_DEBUG(llvm::dbgs().indent(6)
                 << "connect " << render(tgt) << " := " << render(src) << "\n");
      auto defineOp =
          DomainDefineOp::create(builder, wireOp.getLoc(), tgt, src);
      recordDomainDefinition(defineOp);
      return success();
    }
    if (auto *val = dyn_cast<ValueTerm>(term)) {
      auto src = getDomainInScope(builder, domainsInScope, val->value);
      LLVM_DEBUG(llvm::dbgs().indent(6)
                 << "connect " << render(tgt) << " := " << render(src) << "\n");
      auto defineOp =
          DomainDefineOp::create(builder, wireOp.getLoc(), tgt, src);
      recordDomainDefinition(defineOp);
      return success();
    }
    llvm_unreachable("unhandled domain term type");
  }

  if (!isHardware(result) || isColorless(result))
    return success();

  LLVM_DEBUG(llvm::dbgs().indent(4) << "update " << render(wireOp) << "\n");
  OpBuilder builder(wireOp);
  auto *row = getDomainAssociationAsRow(wireOp.getResult());

  SmallVector<Value> domainOperands;
  for (auto [i, element] : llvm::enumerate(
           llvm::map_range(row->elements, [&](auto e) { return find(e); }))) {
    if (auto *val = dyn_cast<ValueTerm>(element)) {
      domainOperands.push_back(
          getDomainInScope(builder, domainsInScope, val->value));
      continue;
    }
    if (auto *var = dyn_cast<VariableTerm>(element)) {
      auto type = DomainType::getFromDomainOp(getDomain(DomainTypeID{i}));
      auto domain =
          solveVarWithAnonDomain(builder, domainsInScope, wireOp, type, var);
      domainOperands.push_back(domain);
      continue;
    }
    assert(0 && "unhandled domain type");
  }
  wireOp.getDomainsMutable().assign(domainOperands);
  return success();
}

LogicalResult ModuleState::updateModuleBody(FModuleOp moduleOp) {
  DenseMap<DomainValue, DomainValue> domainsInScope;

  for (size_t i = 0, e = moduleOp.getNumPorts(); i < e; ++i)
    if (auto port = dyn_cast<DomainValue>(moduleOp.getArgument(i)))
      if (moduleOp.getPortDirection(i) == Direction::In)
        domainsInScope[port] = port;

  auto result = moduleOp.getBodyBlock()->walk([&](Operation *op) -> WalkResult {
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<WireOp>(
            [&](auto wire) { return updateWire(domainsInScope, wire); })
        .Case<FInstanceLike>([&](auto instance) {
          return updateInstance(domainsInScope, instance);
        })
        .Case<DomainCreateOp, DomainCreateAnonOp>([&](auto domain) {
          domainsInScope[domain] = domain;
          return success();
        })
        .Default([&](auto op) { return success(); });
  });
  return failure(result.wasInterrupted());
}

LogicalResult ModuleState::updateModule(FModuleOp moduleOp) {
  auto exports = initializeExportTable(moduleOp);
  PendingUpdates pending;
  getUpdatesForModule(moduleOp, exports, pending);
  applyUpdatesToModule(moduleOp, exports, pending);

  ArrayAttr portDomainInfo;
  if (failed(updateModuleDomainInfo(moduleOp, exports, portDomainInfo)))
    return failure();

  // Record the updated interface change in the update
  auto &entry = getModuleUpdateTable()[moduleOp.getModuleNameAttr()];
  entry.portDomainInfo = portDomainInfo;
  // Keep the complete insertion list available to instances if this module
  // is revisited by the circuit worklist after its interface was generalized.
  // A later analysis visit normally has no new insertions because the module
  // already contains the inferred ports.
  if (!pending.insertions.empty())
    entry.portInsertions = std::move(pending.insertions);

  recordDomainPortAliases(moduleOp);

  LLVM_DEBUG({
    llvm::dbgs().indent(2) << "port summary:\n";
    for (auto port : moduleOp.getBodyBlock()->getArguments()) {
      llvm::dbgs().indent(4) << render(port);
      auto info = cast<ArrayAttr>(
          moduleOp.getDomainInfoAttrForPort(port.getArgNumber()));
      if (info.size()) {
        llvm::dbgs() << " domains [";
        llvm::interleaveComma(
            info.getAsRange<IntegerAttr>(), llvm::dbgs(), [&](auto i) {
              llvm::dbgs() << render(moduleOp.getArgument(i.getUInt()));
            });
        llvm::dbgs() << "]";
      }
      llvm::dbgs() << "\n";
    }
  });

  return success();
}

LogicalResult ModuleState::materializeModule(FModuleOp moduleOp) {
  if (failed(processModule(moduleOp)))
    return failure();

  if (failed(driveModuleOutputDomainPorts(moduleOp))) {
    recordDomainAssignments(moduleOp);
    return failure();
  }

  if (failed(updateModuleBody(moduleOp))) {
    recordDomainAssignments(moduleOp);
    return failure();
  }
  recordDomainAssignments(moduleOp);
  return success();
}

LogicalResult ModuleState::checkModulePorts(FModuleLike moduleOp) {
  auto numDomains = getNumDomains();
  auto domainInfo = moduleOp.getDomainInfoAttr();
  auto numPorts = moduleOp.getNumPorts();

  DenseMap<unsigned, DomainTypeID> domainTypeIDTable;
  for (size_t i = 0; i < numPorts; ++i) {
    if (isa<DomainType>(moduleOp.getPortType(i)))
      domainTypeIDTable[i] = getDomainTypeID(moduleOp, i);
  }

  for (size_t i = 0; i < numPorts; ++i) {
    if (!isHardware(moduleOp.getPortType(i)))
      continue;

    // Record the domain associations of this port.
    SmallVector<IntegerAttr> associations(numDomains);
    for (auto domainPortIndex : getPortDomainAssociation(domainInfo, i)) {
      auto domainTypeID = domainTypeIDTable.at(domainPortIndex.getUInt());
      auto prevDomainPortIndex = associations[domainTypeID.index];
      if (prevDomainPortIndex) {
        emitDuplicatePortDomainError(moduleOp, i, domainTypeID,
                                     prevDomainPortIndex, domainPortIndex);
        return failure();
      }
      associations[domainTypeID.index] = domainPortIndex;
    }

    // Check the associations for completeness.
    for (size_t domainIndex = 0; domainIndex < numDomains; ++domainIndex) {
      auto typeID = DomainTypeID{domainIndex};
      if (!associations[domainIndex]) {
        emitMissingPortDomainAssociationError(moduleOp, typeID, i);
        return failure();
      }
    }
  }

  return success();
}

LogicalResult ModuleState::checkModuleDomainPortDrivers(FModuleOp moduleOp) {
  for (size_t i = 0, e = moduleOp.getNumPorts(); i < e; ++i) {
    auto port = dyn_cast<DomainValue>(moduleOp.getArgument(i));
    if (!port || moduleOp.getPortDirection(i) != Direction::Out ||
        isDriven(port))
      continue;

    auto name = moduleOp.getPortNameAttr(i);
    auto diag = emitError(moduleOp.getPortLocation(i))
                << "undriven domain port " << name;
    noteLocation(diag, moduleOp);
    return failure();
  }

  return success();
}

LogicalResult ModuleState::checkInstanceDomainPortDrivers(FInstanceLike op) {
  for (size_t i = 0, e = op->getNumResults(); i < e; ++i) {
    auto port = dyn_cast<DomainValue>(op->getResult(i));
    if (!port || op.getPortDirection(i) != Direction::In || isDriven(port))
      continue;

    auto name = op.getPortNameAttr(i);
    auto diag = emitError(op.getPortLocation(i))
                << "undriven domain port " << name;
    noteLocation(diag, op);
    return failure();
  }

  return success();
}

LogicalResult ModuleState::checkModuleBody(FModuleOp moduleOp) {
  auto result = moduleOp.getBody().walk([&](FInstanceLike op) -> WalkResult {
    return checkInstanceDomainPortDrivers(op);
  });
  return failure(result.wasInterrupted());
}

LogicalResult ModuleState::inferModule(FModuleOp moduleOp) {
  LLVM_DEBUG(llvm::dbgs() << "infer: " << moduleOp.getModuleName() << "\n");
  if (failed(processModule(moduleOp)))
    return failure();

  if (failed(updateModule(moduleOp))) {
    recordDomainAssignments(moduleOp);
    return failure();
  }
  return success();
}

LogicalResult ModuleState::checkModule(FModuleOp moduleOp) {
  LLVM_DEBUG(llvm::dbgs() << "check: " << moduleOp.getModuleName() << "\n");
  if (failed(checkModulePorts(moduleOp)))
    return failure();

  if (failed(checkModuleDomainPortDrivers(moduleOp)))
    return failure();

  if (failed(checkModuleBody(moduleOp)))
    return failure();

  if (failed(processModule(moduleOp)))
    return failure();
  recordDomainAssignments(moduleOp);
  recordDomainPortAliases(moduleOp);
  return success();
}

LogicalResult ModuleState::checkModule(FExtModuleOp extModuleOp) {
  LLVM_DEBUG(llvm::dbgs() << "check: " << extModuleOp.getModuleName() << "\n");
  return checkModulePorts(extModuleOp);
}

LogicalResult ModuleState::checkAndInferModule(FModuleOp moduleOp) {
  LLVM_DEBUG(llvm::dbgs() << "check/infer: " << moduleOp.getModuleName()
                          << "\n");

  if (failed(checkModulePorts(moduleOp)))
    return failure();

  if (failed(processModule(moduleOp)))
    return failure();

  recordDomainAssignments(moduleOp);
  recordDomainPortAliases(moduleOp);
  return success();
}

//===---------------------------------------------------------------------------
// Domain Stripping.
//===---------------------------------------------------------------------------

/// A helper for stripping domains from a module based on a predicate. The
/// predicate takes a domain name and returns true if that domain should be
/// stripped.
static LogicalResult
stripModuleImpl(FModuleLike op,
                llvm::function_ref<bool(StringAttr)> shouldStripDomain) {
  auto shouldStripType = [&](Type type) {
    if (auto domainType = dyn_cast<DomainType>(type))
      return shouldStripDomain(domainType.getName().getAttr());
    return false;
  };
  WalkResult result = op->walk<mlir::WalkOrder::PostOrder, ReverseIterator>(
      [&](Operation *op) -> WalkResult {
        return TypeSwitch<Operation *, WalkResult>(op)
            .Case<FModuleLike>([&](FModuleLike op) {
              BitVector erasures(op.getNumPorts());
              for (size_t i = 0, e = op.getNumPorts(); i < e; ++i)
                if (shouldStripType(op.getPortType(i)))
                  erasures.set(i);
              if (erasures.any())
                op.erasePorts(erasures);
              return WalkResult::advance();
            })
            .Case<DomainDefineOp>([&](DomainDefineOp op) {
              if (shouldStripType(op.getDest().getType()) ||
                  shouldStripType(op.getSrc().getType()))
                op.erase();
              return WalkResult::advance();
            })
            .Case<DomainCreateOp>([&](DomainCreateOp op) {
              if (shouldStripType(op.getType()))
                op.erase();
              return WalkResult::advance();
            })
            .Case<DomainCreateAnonOp>([&](DomainCreateAnonOp op) {
              if (shouldStripType(op.getType()))
                op.erase();
              return WalkResult::advance();
            })
            .Case<DomainSubfieldOp>([&](DomainSubfieldOp op) {
              // The subfield's result is a property value; decide
              // whether to strip based on the domain it reads from.
              if (shouldStripType(op.getInput().getType())) {
                if (!op->use_empty()) {
                  OpBuilder builder(op);
                  op.replaceAllUsesWith(
                      UnknownValueOp::create(builder, op.getLoc(), op.getType())
                          .getResult());
                }
                op.erase();
              }
              return WalkResult::advance();
            })
            .Case<UnsafeDomainCastOp>([&](UnsafeDomainCastOp op) {
              // Strip cast if any of the domains being cast should be
              // stripped.
              if (llvm::any_of(op.getDomains(), [&](Value domain) {
                    return shouldStripType(domain.getType());
                  })) {
                op.replaceAllUsesWith(op.getInput());
                op.erase();
              }
              return WalkResult::advance();
            })
            .Case<WireOp>([&](WireOp op) {
              // Erase wires of DomainType that should be stripped.
              if (shouldStripType(op.getType(0))) {
                op->erase();
                return WalkResult::advance();
              }
              BitVector erasures(op.getDomains().size());

              // Erase domain operands from regular wires.
              for (int i = 0, e = op.getDomains().size(); i < e; ++i)
                if (shouldStripType(op.getDomains()[i].getType()))
                  erasures.set(i);

              op->eraseOperands(erasures);
              return WalkResult::advance();
            })
            .Case<FInstanceLike>([&](auto op) {
              auto n = op.getNumPorts();
              BitVector erasures(n);
              for (size_t i = 0; i < n; ++i)
                if (shouldStripType(op->getResult(i).getType()))
                  erasures.set(i);
              if (erasures.any()) {
                op.cloneWithErasedPortsAndReplaceUses(erasures);
                op.erase();
              }
              return WalkResult::advance();
            })
            .Default([&](Operation *op) {
              // All operations that can have DomainType are handled
              // above. If we encounter one here, it's a bug in the IR
              // or this pass.
              for (auto type :
                   concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
                if (isa<DomainType>(type)) {
                  op->emitOpError("cannot be stripped");
                  return WalkResult::interrupt();
                }
              }
              return WalkResult::advance();
            });
      });
  return failure(result.wasInterrupted());
}

static LogicalResult stripDomainsFromCircuit(
    MLIRContext *context, CircuitOp circuit,
    llvm::function_ref<bool(StringAttr)> shouldStripDomain) {
  // Collect modules and erase matching DomainOp declarations.
  llvm::SmallVector<FModuleLike> modules;
  for (Operation &op : make_early_inc_range(*circuit.getBodyBlock())) {
    TypeSwitch<Operation *, void>(&op)
        .Case<FModuleLike>([&](FModuleLike op) { modules.push_back(op); })
        .Case<DomainOp>([&](DomainOp op) {
          // Erase domain declaration if its name should be stripped.
          if (shouldStripDomain(op.getNameAttr()))
            op.erase();
        });
  }

  // Strip domains from all modules in parallel.
  return failableParallelForEach(context, modules, [&](FModuleLike module) {
    return stripModuleImpl(module, shouldStripDomain);
  });
}

//===---------------------------------------------------------------------------
// InferDomainsPass: Top-level pass implementation.
//===---------------------------------------------------------------------------

LogicalResult CircuitState::runOnModule(Operation *op) {
  assert(mode != InferDomainsMode::Strip);
  ModuleState state(*this);
  if (auto moduleOp = dyn_cast<FModuleOp>(op)) {
    if (mode == InferDomainsMode::Check)
      return state.checkModule(moduleOp);

    if (mode == InferDomainsMode::InferAll || moduleOp.isPrivate())
      return state.inferModule(moduleOp);

    return state.checkAndInferModule(moduleOp);
  }

  if (auto extModuleOp = dyn_cast<FExtModuleOp>(op))
    return state.checkModule(extModuleOp);

  return success();
}

LogicalResult CircuitState::materializeOnModule(Operation *op) {
  assert(mode != InferDomainsMode::Strip && mode != InferDomainsMode::Check);
  ModuleState state(*this);
  if (auto moduleOp = dyn_cast<FModuleOp>(op))
    return state.materializeModule(moduleOp);
  return success();
}

LogicalResult CircuitState::run() {
  DenseSet<Operation *> errored;
  SmallVector<igraph::InstanceGraphNode *> worklist;
  DenseSet<igraph::InstanceGraphNode *> queued;

  // Seed the worklist in dependency order. Interface and alias summaries
  // are published as each module is processed; a changed summary requeues
  // all of the module's users below.  This is important when a module is
  // reached through more than one hierarchy path, or when a graph edge is
  // revisited after an interface update.
  instanceGraph.walkPostOrder([&](auto &node) {
    worklist.push_back(&node);
    queued.insert(&node);
  });

  for (size_t workIndex = 0; workIndex < worklist.size(); ++workIndex) {
    auto *node = worklist[workIndex];
    queued.erase(node);
    auto moduleOp = node->getModule();
    bool dependencyFailed = false;
    for (auto *inst : *node) {
      if (errored.contains(inst->getTarget()->getModule())) {
        errored.insert(moduleOp);
        dependencyFailed = true;
        break;
      }
    }
    if (dependencyFailed)
      continue;

    size_t oldNumPorts = 0;
    Attribute oldDomainInfo;
    if (auto moduleLike = dyn_cast<FModuleLike>(moduleOp.getOperation())) {
      oldNumPorts = moduleLike.getNumPorts();
      oldDomainInfo = moduleLike.getDomainInfoAttr();
    }

    auto oldAliasesIt =
        moduleDomainPortAliases.find(moduleOp.getModuleNameAttr());
    bool hadAliases = oldAliasesIt != moduleDomainPortAliases.end();
    DomainPortAliases oldAliases;
    if (hadAliases)
      oldAliases = oldAliasesIt->second;

    if (failed(runOnModule(node->getModule())))
      errored.insert(moduleOp);

    if (errored.contains(moduleOp))
      continue;

    bool interfaceChanged = false;
    if (auto moduleLike = dyn_cast<FModuleLike>(moduleOp.getOperation()))
      interfaceChanged = oldNumPorts != moduleLike.getNumPorts() ||
                         oldDomainInfo != moduleLike.getDomainInfoAttr();

    auto newAliasesIt =
        moduleDomainPortAliases.find(moduleOp.getModuleNameAttr());
    bool aliasesChanged =
        hadAliases != (newAliasesIt != moduleDomainPortAliases.end());
    if (!aliasesChanged && hadAliases)
      aliasesChanged = oldAliases != newAliasesIt->second;

    if (interfaceChanged || aliasesChanged) {
      for (auto *use : node->uses()) {
        auto *user = use->getParent();
        if (queued.insert(user).second)
          worklist.push_back(user);
      }
    }
  }
  if (!errored.empty()) {
    if (shouldEmitDomainReport() && failed(writeDomainReport(false)))
      return failure();
    return failure();
  }

  // The analysis above deliberately leaves body-generated domain operations
  // out of the IR. Materialize them only once all effective module interfaces
  // and alias summaries are stable, so a worklist revisit cannot duplicate
  // them. Definitions for newly inserted output ports are kept during
  // analysis because they establish terms needed by later visits.
  if (mode == InferDomainsMode::Check)
    return shouldEmitDomainReport() ? writeDomainReport(true) : success();

  bool materializationFailed = false;
  instanceGraph.walkPostOrder([&](auto &node) {
    if (failed(materializeOnModule(node.getModule())))
      materializationFailed = true;
  });
  if (materializationFailed) {
    if (shouldEmitDomainReport() && failed(writeDomainReport(false)))
      return failure();
    return failure();
  }
  return shouldEmitDomainReport() ? writeDomainReport(true) : success();
}

namespace {
struct InferDomainsPass
    : public circt::firrtl::impl::InferDomainsBase<InferDomainsPass> {
  using Base::Base;
  void runOnOperation() override {
    CIRCT_DEBUG_SCOPED_PASS_LOGGER(this);
    auto circuit = getOperation();

    if (mode == InferDomainsMode::Strip) {
      if (!reportJson.empty()) {
        circuit.emitError() << "domain report requires domain inference or "
                               "checking to be enabled";
        return signalPassFailure();
      }
      // Strip all domain types
      if (failed(stripDomainsFromCircuit(&getContext(), circuit,
                                         [](StringAttr) { return true; })))
        signalPassFailure();
      return;
    }

    // Strip skipped domains in a prepass before checking/inference
    if (!skippedDomains.empty()) {
      DenseSet<StringAttr> skippedNames;
      auto *context = &getContext();
      for (const auto &name : skippedDomains)
        skippedNames.insert(StringAttr::get(context, name));

      if (failed(
              stripDomainsFromCircuit(context, circuit, [&](StringAttr name) {
                return skippedNames.contains(name);
              })))
        return signalPassFailure();
    }

    auto &instanceGraph = getAnalysis<InstanceGraph>();
    auto &symbolTable = getAnalysis<SymbolTable>();
    auto &innerSymbolTableCollection =
        getAnalysis<InnerSymbolTableCollection>();
    circt::hw::InnerRefNamespace innerRefNamespace{symbolTable,
                                                   innerSymbolTableCollection};
    CircuitState state(circuit, instanceGraph, innerRefNamespace, mode,
                       reportJson);
    if (failed(state.run()))
      signalPassFailure();
  }
};
} // namespace
