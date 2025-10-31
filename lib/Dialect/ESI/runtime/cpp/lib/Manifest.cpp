//===- Manifest.cpp - Metadata on the accelerator -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT (lib/dialect/ESI/runtime/cpp/).
//
//===----------------------------------------------------------------------===//

#include "esi/Manifest.h"
#include "esi/Accelerator.h"
#include "esi/Services.h"

#include <nlohmann/json.hpp>
#include <sstream>

using namespace ::esi;
using ServiceTable = AcceleratorConnection::ServiceTable;

// This is a proxy class to the manifest JSON. It is used to avoid having to
// include the JSON parser in the header. Forward references don't work since
// nlohmann::json is a rather complex template.
//
// Plus, it allows us to hide some implementation functions from the header
// file.
class Manifest::Impl {
  friend class ::esi::Manifest;

public:
  Impl(Context &ctxt, const std::string &jsonManifest);

  auto at(const std::string &key) const { return manifestJson.at(key); }

  // Get the module info (if any) for the module instance in 'json'.
  std::optional<ModuleInfo> getModInfo(const nlohmann::json &) const;

  /// Go through the "service_decls" section of the manifest and populate the
  /// services table as appropriate.
  void scanServiceDecls(AcceleratorConnection &, const nlohmann::json &,
                        ServiceTable &) const;

  void createEngine(AcceleratorConnection &, AppIDPath appID,
                    const nlohmann::json &) const;

  /// Get a Service for the service specified in 'json'. Update the
  /// activeServices table. TODO: re-using this for the engines section is a
  /// terrible hack. Figure out a better way.
  services::Service *getService(AppIDPath idPath, AcceleratorConnection &,
                                const nlohmann::json &,
                                ServiceTable &activeServices,
                                bool isEngine = false) const;

  /// Get all the services in the description of an instance. Update the active
  /// services table.
  std::vector<services::Service *>
  getServices(AppIDPath idPath, AcceleratorConnection &, const nlohmann::json &,
              ServiceTable &activeServices) const;

  /// Get the bundle ports for the instance at 'idPath' and specified in
  /// 'instJson'. Look them up in 'activeServies'.
  std::vector<std::unique_ptr<BundlePort>>
  getBundlePorts(AcceleratorConnection &acc, AppIDPath idPath,
                 const ServiceTable &activeServices,
                 const nlohmann::json &instJson) const;

  /// Build the set of child instances (recursively) for the module instance
  /// description.
  std::vector<std::unique_ptr<Instance>>
  getChildInstances(AppIDPath idPath, AcceleratorConnection &acc,
                    const ServiceTable &activeServices,
                    const nlohmann::json &instJson) const;

  /// Get a single child instance. Implicitly copy the active services table so
  /// that it can be safely updated for the child's branch of the tree.
  std::unique_ptr<Instance>
  getChildInstance(AppIDPath idPath, AcceleratorConnection &acc,
                   ServiceTable activeServices,
                   const nlohmann::json &childJson) const;

  /// Parse all the types and populate the types table.
  void populateTypes(const nlohmann::json &typesJson);

  /// Get the ordered list of types from the manifest.
  const std::vector<const Type *> &getTypeTable() const { return _typeTable; }

  /// Build a dynamic API for the Accelerator connection 'acc' based on the
  /// manifest stored herein.
  std::unique_ptr<Accelerator>
  buildAccelerator(AcceleratorConnection &acc) const;

  const Type *parseType(const nlohmann::json &typeJson);

  const std::map<std::string, const ModuleInfo> &getSymbolInfo() const {
    return symbolInfoCache;
  }

private:
  Context &ctxt;
  std::vector<const Type *> _typeTable;

  std::optional<const Type *> getType(Type::ID id) const {
    return ctxt.getType(id);
  }

  std::any getAny(const nlohmann::json &value) const;
  void parseModuleMetadata(ModuleInfo &info, const nlohmann::json &mod) const;
  void parseModuleConsts(ModuleInfo &info, const nlohmann::json &mod) const;

  // The parsed json.
  nlohmann::json manifestJson;
  // Cache the module info for each symbol.
  std::map<std::string, const ModuleInfo> symbolInfoCache;
};

//===----------------------------------------------------------------------===//
// Simple JSON -> object parsers.
//===----------------------------------------------------------------------===//

static std::optional<AppID> parseID(const nlohmann::json &jsonID) {
  if (!jsonID.is_object())
    return std::nullopt;
  std::optional<uint32_t> idx;
  if (jsonID.contains("index"))
    idx = jsonID.at("index").get<uint32_t>();
  if (jsonID.contains("name") && jsonID.size() <= 2)
    return AppID(jsonID.at("name").get<std::string>(), idx);
  return std::nullopt;
}

static AppID parseIDChecked(const nlohmann::json &jsonID) {
  std::optional<AppID> id = parseID(jsonID);
  if (!id)
    throw std::runtime_error("Malformed manifest: invalid appID");
  return *id;
}
static AppIDPath parseIDPath(const nlohmann::json &jsonIDPath) {
  AppIDPath ret;
  for (auto &idJson : jsonIDPath)
    ret.push_back(parseIDChecked(idJson));
  return ret;
}

static ServicePortDesc parseServicePort(const nlohmann::json &jsonPort) {
  return ServicePortDesc{jsonPort.at("serviceName").get<std::string>(),
                         jsonPort.at("port").get<std::string>()};
}

/// Convert the json value to a 'std::any', which can be exposed outside of this
/// file.
std::any Manifest::Impl::getAny(const nlohmann::json &value) const {
  auto getObject = [this](const nlohmann::json &json) -> std::any {
    std::map<std::string, std::any> ret;
    for (auto &e : json.items())
      ret[e.key()] = getAny(e.value());

    // If this can be converted to a constant, do so.
    if (ret.size() != 2 || !ret.contains("type") || !ret.contains("value"))
      return ret;
    std::any value = ret.at("value");
    std::any typeID = ret.at("type");
    if (typeID.type() != typeid(std::string))
      return ret;
    std::optional<const Type *> type =
        getType(std::any_cast<std::string>(type));
    if (!type)
      return ret;
    // TODO: Check or guide the conversion of the value to the type based on the
    // type.
    return Constant{value, type};
  };

  auto getArray = [this](const nlohmann::json &json) -> std::any {
    std::vector<std::any> ret;
    for (auto &e : json)
      ret.push_back(getAny(e));
    return ret;
  };

  auto getValue = [&](const nlohmann::json &innerValue) -> std::any {
    if (innerValue.is_string())
      return innerValue.get<std::string>();
    else if (innerValue.is_number_unsigned())
      return innerValue.get<uint64_t>();
    else if (innerValue.is_number_integer())
      return innerValue.get<int64_t>();
    else if (innerValue.is_number_float())
      return innerValue.get<double>();
    else if (innerValue.is_boolean())
      return innerValue.get<bool>();
    else if (innerValue.is_null())
      return innerValue.get<std::nullptr_t>();
    else if (innerValue.is_object())
      return getObject(innerValue);
    else if (innerValue.is_array())
      return getArray(innerValue);
    else
      throw std::runtime_error("Unknown type in manifest: " +
                               innerValue.dump(2));
  };

  std::optional<AppID> appid = parseID(value);
  if (appid)
    return *appid;
  if (!value.is_object() || !value.contains("type") || !value.contains("value"))
    return getValue(value);
  return Constant{getValue(value.at("value")), getType(value.at("type"))};
}

void Manifest::Impl::parseModuleMetadata(ModuleInfo &info,
                                         const nlohmann::json &mod) const {
  for (auto &extra : mod.items())
    if (extra.key() != "name" && extra.key() != "summary" &&
        extra.key() != "version" && extra.key() != "repo" &&
        extra.key() != "commitHash")
      info.extra[extra.key()] = getAny(extra.value());

  auto value = [&](const std::string &key) -> std::optional<std::string> {
    auto f = mod.find(key);
    if (f == mod.end())
      return std::nullopt;
    return f.value();
  };
  info.name = value("name");
  info.summary = value("summary");
  info.version = value("version");
  info.repo = value("repo");
  info.commitHash = value("commitHash");
}

void Manifest::Impl::parseModuleConsts(ModuleInfo &info,
                                       const nlohmann::json &mod) const {
  for (auto &item : mod.items()) {
    std::any value = getAny(item.value());
    auto *c = std::any_cast<Constant>(&value);
    if (c)
      info.constants[item.key()] = *c;
    else
      // If the value isn't a "proper" constant, present it as one with no type.
      info.constants[item.key()] = Constant{value, std::nullopt};
  }
}

//===----------------------------------------------------------------------===//
// Manifest::Impl class implementation.
//===----------------------------------------------------------------------===//

Manifest::Impl::Impl(Context &ctxt, const std::string &manifestStr)
    : ctxt(ctxt) {
  manifestJson = nlohmann::ordered_json::parse(manifestStr);

  try {
    // Populate the types table first since anything else might need it.
    populateTypes(manifestJson.at("types"));

    // Populate the symbol info cache.
    for (auto &mod : manifestJson.at("modules")) {
      ModuleInfo info;
      if (mod.contains("symInfo"))
        parseModuleMetadata(info, mod.at("symInfo"));
      if (mod.contains("symConsts"))
        parseModuleConsts(info, mod.at("symConsts"));
      symbolInfoCache.insert(make_pair(mod.at("symbol"), info));
    }
  } catch (const std::exception &e) {
    std::string msg = "malformed manifest: " + std::string(e.what());
    if (manifestJson.at("apiVersion") == 0)
      msg += " (schema version 0 is not considered stable)";
    throw std::runtime_error(msg);
  }
}

std::unique_ptr<Accelerator>
Manifest::Impl::buildAccelerator(AcceleratorConnection &acc) const {
  ServiceTable activeSvcs;

  auto designJson = manifestJson.at("design");

  // Create all of the engines at the top level of the design.
  // TODO: support engines at lower levels.
  auto enginesIter = designJson.find("engines");
  if (enginesIter != designJson.end())
    for (auto &engineDesc : enginesIter.value())
      createEngine(acc, {}, engineDesc);

  // Get the initial active services table. Update it as we descend down.
  auto svcDecls = manifestJson.at("serviceDeclarations");
  scanServiceDecls(acc, svcDecls, activeSvcs);

  // Get the services instantiated at the top level.
  std::vector<services::Service *> services =
      getServices({}, acc, designJson, activeSvcs);

  // Get the ports at the top level.
  auto ports = getBundlePorts(acc, {}, activeSvcs, designJson);

  return std::make_unique<Accelerator>(
      getModInfo(designJson),
      getChildInstances({}, acc, activeSvcs, designJson), services, ports);
}

std::optional<ModuleInfo>
Manifest::Impl::getModInfo(const nlohmann::json &json) const {
  auto instOfIter = json.find("instOf");
  if (instOfIter == json.end())
    return std::nullopt;
  auto f = symbolInfoCache.find(instOfIter.value());
  if (f != symbolInfoCache.end())
    return f->second;
  return std::nullopt;
}

/// TODO: Hack. This method is a giant hack to reuse the getService method for
/// engines. It works, but it ain't pretty and it ain't right.
void Manifest::Impl::createEngine(AcceleratorConnection &acc, AppIDPath idPath,
                                  const nlohmann::json &eng) const {
  ServiceTable dummy;
  getService(idPath, acc, eng, dummy, /*isEngine=*/true);
}

void Manifest::Impl::scanServiceDecls(AcceleratorConnection &acc,
                                      const nlohmann::json &svcDecls,
                                      ServiceTable &activeServices) const {
  for (auto &svcDecl : svcDecls) {
    // Get the implementation details.
    ServiceImplDetails svcDetails;
    for (auto &detail : svcDecl.items())
      svcDetails[detail.key()] = getAny(detail.value());

    // Create the service.
    auto serviceNameIter = svcDecl.find("serviceName");
    std::string serviceName;
    if (serviceNameIter != svcDecl.end())
      serviceName = serviceNameIter.value();
    services::Service::Type svcId =
        services::ServiceRegistry::lookupServiceType(serviceName);
    auto svc = acc.getService(svcId, /*id=*/{}, /*implName=*/"",
                              /*details=*/svcDetails, /*clients=*/{});
    if (svc)
      activeServices[svcDecl.at("symbol")] = svc;
  }
}

std::vector<std::unique_ptr<Instance>>
Manifest::Impl::getChildInstances(AppIDPath idPath, AcceleratorConnection &acc,
                                  const ServiceTable &activeServices,
                                  const nlohmann::json &instJson) const {
  std::vector<std::unique_ptr<Instance>> ret;
  auto childrenIter = instJson.find("children");
  if (childrenIter == instJson.end())
    return ret;
  for (auto &child : childrenIter.value())
    ret.emplace_back(getChildInstance(idPath, acc, activeServices, child));
  return ret;
}

std::unique_ptr<Instance>
Manifest::Impl::getChildInstance(AppIDPath idPath, AcceleratorConnection &acc,
                                 ServiceTable activeServices,
                                 const nlohmann::json &child) const {
  AppID childID = parseIDChecked(child.at("appID"));
  idPath.push_back(childID);

  std::vector<services::Service *> services =
      getServices(idPath, acc, child, activeServices);

  auto children = getChildInstances(idPath, acc, activeServices, child);
  auto ports = getBundlePorts(acc, idPath, activeServices, child);
  return std::make_unique<Instance>(parseIDChecked(child.at("appID")),
                                    getModInfo(child), std::move(children),
                                    services, ports);
}

services::Service *Manifest::Impl::getService(AppIDPath idPath,
                                              AcceleratorConnection &acc,
                                              const nlohmann::json &svcJson,
                                              ServiceTable &activeServices,
                                              bool isEngine) const {

  AppID id = parseIDChecked(svcJson.at("appID"));
  idPath.push_back(id);

  // Get all the client info, including the implementation details.
  HWClientDetails clientDetails;
  for (auto &client : svcJson.at("clientDetails")) {
    HWClientDetail clientDetail;
    for (auto &detail : client.items()) {
      if (detail.key() == "relAppIDPath")
        clientDetail.relPath = parseIDPath(detail.value());
      else if (detail.key() == "servicePort")
        clientDetail.port = parseServicePort(detail.value());
      else if (detail.key() == "channelAssignments") {
        for (auto &chan : detail.value().items()) {
          ChannelAssignment chanAssign;
          for (auto &assign : chan.value().items())
            if (assign.key() == "type")
              chanAssign.type = assign.value();
            else
              chanAssign.implOptions[assign.key()] = getAny(assign.value());
          clientDetail.channelAssignments[chan.key()] = chanAssign;
        }
      } else
        clientDetail.implOptions[detail.key()] = getAny(detail.value());
    }
    clientDetails.push_back(clientDetail);
  }

  // Get the implementation details.
  ServiceImplDetails svcDetails;
  std::string implName;
  std::string service;
  for (auto &detail : svcJson.items()) {
    if (detail.key() == "appID" || detail.key() == "clientDetails")
      continue;
    if (detail.key() == "serviceImplName")
      implName = detail.value();
    else if (detail.key() == "service")
      service = detail.value().get<std::string>();
    else
      svcDetails[detail.key()] = getAny(detail.value());
  }

  // Create the service.
  services::Service *svc = nullptr;
  auto activeServiceIter = activeServices.find(service);
  if (activeServiceIter != activeServices.end()) {
    services::Service::Type svcType =
        services::ServiceRegistry::lookupServiceType(
            activeServiceIter->second->getServiceSymbol());
    svc = activeServiceIter->second->getChildService(svcType, idPath, implName,
                                                     svcDetails, clientDetails);
  } else {
    services::Service::Type svcType =
        services::ServiceRegistry::lookupServiceType(service);
    if (isEngine)
      acc.createEngine(implName, idPath, svcDetails, clientDetails);
    else
      svc =
          acc.getService(svcType, idPath, implName, svcDetails, clientDetails);
  }

  if (svc)
    // Update the active services table.
    activeServices[service] = svc;
  return svc;
}

std::vector<services::Service *>
Manifest::Impl::getServices(AppIDPath idPath, AcceleratorConnection &acc,
                            const nlohmann::json &svcsJson,
                            ServiceTable &activeServices) const {
  std::vector<services::Service *> ret;
  auto svcsIter = svcsJson.find("services");
  if (svcsIter == svcsJson.end())
    return ret;

  for (auto &svc : svcsIter.value())
    ret.emplace_back(getService(idPath, acc, svc, activeServices));
  return ret;
}

std::vector<std::unique_ptr<BundlePort>>
Manifest::Impl::getBundlePorts(AcceleratorConnection &acc, AppIDPath idPath,
                               const ServiceTable &activeServices,
                               const nlohmann::json &instJson) const {
  std::vector<std::unique_ptr<BundlePort>> ret;
  auto clientPortsIter = instJson.find("clientPorts");
  if (clientPortsIter == instJson.end())
    return ret;

  for (auto &content : clientPortsIter.value()) {
    // Lookup the requested service in the active services table.
    std::string serviceName = "";
    if (auto f = content.find("servicePort"); f != content.end())
      serviceName = parseServicePort(f.value()).name;
    auto svcIter = activeServices.find(serviceName);
    if (svcIter == activeServices.end()) {
      // If a specific service isn't found, search for the default service
      // (typically provided by a BSP).
      if (svcIter = activeServices.find(""); svcIter == activeServices.end())
        throw std::runtime_error(
            "Malformed manifest: could not find active service '" +
            serviceName + "'");
    }
    services::Service *svc = svcIter->second;

    std::string typeName = content.at("typeID");
    auto type = getType(typeName);
    if (!type)
      throw std::runtime_error(
          "Malformed manifest: could not find port type '" + typeName + "'");
    const BundleType *bundleType = dynamic_cast<const BundleType *>(*type);
    if (!bundleType)
      throw std::runtime_error("Malformed manifest: type '" + typeName +
                               "' is not a bundle type");

    idPath.push_back(parseIDChecked(content.at("appID")));

    BundlePort *svcPort = svc->getPort(idPath, bundleType);
    if (svcPort)
      ret.emplace_back(svcPort);
    // Since we share idPath between iterations, pop the last element before the
    // next iteration.
    idPath.pop_back();
  }
  return ret;
}

namespace {
const Type *parseType(const nlohmann::json &typeJson, Context &ctxt);

BundleType *parseBundleType(const nlohmann::json &typeJson, Context &cache) {
  assert(typeJson.at("mnemonic") == "bundle");

  std::vector<std::tuple<std::string, BundleType::Direction, const Type *>>
      channels;
  for (auto &chanJson : typeJson["channels"]) {
    std::string dirStr = chanJson.at("direction");
    BundleType::Direction dir;
    if (dirStr == "to")
      dir = BundleType::Direction::To;
    else if (dirStr == "from")
      dir = BundleType::Direction::From;
    else
      throw std::runtime_error("Malformed manifest: unknown direction '" +
                               dirStr + "'");
    channels.emplace_back(chanJson.at("name"), dir,
                          parseType(chanJson["type"], cache));
  }
  return new BundleType(typeJson.at("id"), channels);
}

ChannelType *parseChannelType(const nlohmann::json &typeJson, Context &cache) {
  assert(typeJson.at("mnemonic") == "channel");
  return new ChannelType(typeJson.at("id"),
                         parseType(typeJson.at("inner"), cache));
}

Type *parseInt(const nlohmann::json &typeJson, Context &cache) {
  assert(typeJson.at("mnemonic") == "int");
  std::string sign = typeJson.at("signedness");
  uint64_t width = typeJson.at("hwBitwidth");
  Type::ID id = typeJson.at("id");

  if (sign == "signed")
    return new SIntType(id, width);
  else if (sign == "unsigned")
    return new UIntType(id, width);
  else if (sign == "signless" && width == 0)
    // By convention, a zero-width signless integer is a void type.
    return new VoidType(id);
  else if (sign == "signless" && width > 0)
    return new BitsType(id, width);
  else
    throw std::runtime_error("Malformed manifest: unknown sign '" + sign + "'");
}

StructType *parseStruct(const nlohmann::json &typeJson, Context &cache) {
  assert(typeJson.at("mnemonic") == "struct");
  std::vector<std::pair<std::string, const Type *>> fields;
  for (auto &fieldJson : typeJson["fields"])
    fields.emplace_back(fieldJson.at("name"),
                        parseType(fieldJson["type"], cache));
  return new StructType(typeJson.at("id"), fields);
}

ArrayType *parseArray(const nlohmann::json &typeJson, Context &cache) {
  assert(typeJson.at("mnemonic") == "array");
  uint64_t size = typeJson.at("size");
  return new ArrayType(typeJson.at("id"),
                       parseType(typeJson.at("element"), cache), size);
}

using TypeParser = std::function<Type *(const nlohmann::json &, Context &)>;
const std::map<std::string_view, TypeParser> typeParsers = {
    {"bundle", parseBundleType},
    {"channel", parseChannelType},
    {"std::any", [](const nlohmann::json &typeJson,
                    Context &cache) { return new AnyType(typeJson.at("id")); }},
    {"int", parseInt},
    {"struct", parseStruct},
    {"array", parseArray},

};

// Parse a type if it doesn't already exist in the cache.
const Type *parseType(const nlohmann::json &typeJson, Context &cache) {
  std::string id;
  if (typeJson.is_string())
    id = typeJson.get<std::string>();
  else
    id = typeJson.at("id");
  if (std::optional<const Type *> t = cache.getType(id))
    return *t;
  if (typeJson.is_string())
    throw std::runtime_error("malformed manifest: unknown type '" + id + "'");

  Type *t;
  std::string mnemonic = typeJson.at("mnemonic");
  auto f = typeParsers.find(mnemonic);
  if (f != typeParsers.end())
    t = f->second(typeJson, cache);
  else
    // Types we don't know about are opaque.
    t = new Type(id);

  // Insert into the cache.
  cache.registerType(t);
  return t;
}
} // namespace

const Type *Manifest::Impl::parseType(const nlohmann::json &typeJson) {
  return ::parseType(typeJson, ctxt);
}

void Manifest::Impl::populateTypes(const nlohmann::json &typesJson) {
  for (auto &typeJson : typesJson)
    _typeTable.push_back(parseType(typeJson));
}

//===----------------------------------------------------------------------===//
// Manifest class implementation.
//===----------------------------------------------------------------------===//

Manifest::Manifest(Context &ctxt, const std::string &jsonManifest)
    : impl(new Impl(ctxt, jsonManifest)) {}

Manifest::~Manifest() { delete impl; }

uint32_t Manifest::getApiVersion() const {
  return impl->at("apiVersion").get<uint32_t>();
}

std::vector<ModuleInfo> Manifest::getModuleInfos() const {
  std::vector<ModuleInfo> ret;
  for (auto &[symbol, info] : impl->getSymbolInfo())
    ret.push_back(info);
  return ret;
}

Accelerator *Manifest::buildAccelerator(AcceleratorConnection &acc) const {
  try {
    return acc.takeOwnership(impl->buildAccelerator(acc));
  } catch (const std::exception &e) {
    std::string msg = "malformed manifest: " + std::string(e.what());
    if (getApiVersion() == 0)
      msg += " (schema version 0 is not considered stable)";
    throw std::runtime_error(msg);
  }
}

const std::vector<const Type *> &Manifest::getTypeTable() const {
  return impl->getTypeTable();
}

//===----------------------------------------------------------------------===//
// POCO helpers.
//===----------------------------------------------------------------------===//

// Print a module info, including the extra metadata.
std::ostream &operator<<(std::ostream &os, const ModuleInfo &m) {
  auto printAny = [&os](std::any a) {
    if (std::any_cast<Constant>(&a))
      a = std::any_cast<Constant>(a).value;

    const std::type_info &t = a.type();
    if (t == typeid(std::string))
      os << std::any_cast<std::string>(a);
    else if (t == typeid(int64_t))
      os << std::any_cast<int64_t>(a);
    else if (t == typeid(uint64_t))
      os << std::any_cast<uint64_t>(a);
    else if (t == typeid(double))
      os << std::any_cast<double>(a);
    else if (t == typeid(bool))
      os << std::any_cast<bool>(a);
    else if (t == typeid(std::nullptr_t))
      os << "null";
    else
      os << "unknown";
  };

  if (m.name)
    os << *m.name << " ";
  if (m.version)
    os << *m.version << " ";
  if (m.name || m.version)
    os << std::endl;
  if (m.repo || m.commitHash) {
    os << "  Version control: ";
    if (m.repo)
      os << *m.repo;
    if (m.commitHash)
      os << "@" << *m.commitHash;
    os << std::endl;
  }
  if (m.summary)
    os << "  " << *m.summary;
  os << "\n";

  if (!m.constants.empty()) {
    os << "  Constants:\n";
    for (auto &e : m.constants) {
      os << "    " << e.first << ": ";
      printAny(e.second);
      os << "\n";
    }
  }

  if (!m.extra.empty()) {
    os << "  Extra metadata:\n";
    for (auto &e : m.extra) {
      os << "    " << e.first << ": ";
      printAny(e.second);
      os << "\n";
    }
  }
  return os;
}

namespace esi {
AppIDPath AppIDPath::operator+(const AppIDPath &b) const {
  AppIDPath ret = *this;
  ret.insert(ret.end(), b.begin(), b.end());
  return ret;
}

AppIDPath AppIDPath::parent() const {
  AppIDPath ret = *this;
  if (!ret.empty())
    ret.pop_back();
  return ret;
}

std::string AppIDPath::toStr() const {
  std::ostringstream os;
  os << *this;
  return os.str();
}

bool operator<(const AppID &a, const AppID &b) {
  if (a.name != b.name)
    return a.name < b.name;
  return a.idx < b.idx;
}
bool operator<(const AppIDPath &a, const AppIDPath &b) {
  if (a.size() != b.size())
    return a.size() < b.size();
  for (size_t i = 0, e = a.size(); i < e; ++i)
    if (a[i] != b[i])
      return a[i] < b[i];
  return false;
}

std::ostream &operator<<(std::ostream &os, const AppID &id) {
  os << id.name;
  if (id.idx)
    os << "[" << *id.idx << "]";
  return os;
}
std::ostream &operator<<(std::ostream &os, const AppIDPath &path) {
  for (size_t i = 0, e = path.size(); i < e; ++i) {
    if (i > 0)
      os << '.';
    os << path[i];
  }
  return os;
}

} // namespace esi
