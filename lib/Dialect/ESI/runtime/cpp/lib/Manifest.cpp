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

using namespace std;

using namespace esi;

// While building the design, keep around a map of active services indexed by
// the service name. When a new service is encountered during descent, add it to
// the table (perhaps overwriting one). Modifications to the table only apply to
// the current branch, so copy this and update it at each level of the tree.
using ServiceTable = map<string, services::Service *>;

// This is a proxy class to the manifest JSON. It is used to avoid having to
// include the JSON parser in the header. Forward references don't work since
// nlohmann::json is a rather complex template.
//
// Plus, it allows us to hide some implementation functions from the header
// file.
class Manifest::Impl {
  friend class ::esi::Manifest;

public:
  Impl(Context &ctxt, const string &jsonManifest);

  auto at(const string &key) const { return manifestJson.at(key); }

  // Get the module info (if any) for the module instance in 'json'.
  optional<ModuleInfo> getModInfo(const nlohmann::json &) const;

  /// Go through the "service_decls" section of the manifest and populate the
  /// services table as appropriate.
  void scanServiceDecls(AcceleratorConnection &, const nlohmann::json &,
                        ServiceTable &) const;

  /// Get a Service for the service specified in 'json'. Update the
  /// activeServices table.
  services::Service *getService(AppIDPath idPath, AcceleratorConnection &,
                                const nlohmann::json &,
                                ServiceTable &activeServices) const;

  /// Get all the services in the description of an instance. Update the active
  /// services table.
  vector<services::Service *> getServices(AppIDPath idPath,
                                          AcceleratorConnection &,
                                          const nlohmann::json &,
                                          ServiceTable &activeServices) const;

  /// Get the bundle ports for the instance at 'idPath' and specified in
  /// 'instJson'. Look them up in 'activeServies'.
  vector<std::unique_ptr<BundlePort>>
  getBundlePorts(AcceleratorConnection &acc, AppIDPath idPath,
                 const ServiceTable &activeServices,
                 const nlohmann::json &instJson) const;

  /// Build the set of child instances (recursively) for the module instance
  /// description.
  vector<unique_ptr<Instance>>
  getChildInstances(AppIDPath idPath, AcceleratorConnection &acc,
                    const ServiceTable &activeServices,
                    const nlohmann::json &instJson) const;

  /// Get a single child instance. Implicitly copy the active services table so
  /// that it can be safely updated for the child's branch of the tree.
  unique_ptr<Instance> getChildInstance(AppIDPath idPath,
                                        AcceleratorConnection &acc,
                                        ServiceTable activeServices,
                                        const nlohmann::json &childJson) const;

  /// Parse all the types and populate the types table.
  void populateTypes(const nlohmann::json &typesJson);

  /// Get the ordered list of types from the manifest.
  const vector<const Type *> &getTypeTable() const { return _typeTable; }

  /// Build a dynamic API for the Accelerator connection 'acc' based on the
  /// manifest stored herein.
  unique_ptr<Accelerator> buildAccelerator(AcceleratorConnection &acc) const;

  const Type *parseType(const nlohmann::json &typeJson);

private:
  Context &ctxt;
  vector<const Type *> _typeTable;

  optional<const Type *> getType(Type::ID id) const { return ctxt.getType(id); }

  // The parsed json.
  nlohmann::json manifestJson;
  // Cache the module info for each symbol.
  map<string, ModuleInfo> symbolInfoCache;
};

//===----------------------------------------------------------------------===//
// Simple JSON -> object parsers.
//===----------------------------------------------------------------------===//

static AppID parseID(const nlohmann::json &jsonID) {
  optional<uint32_t> idx;
  if (jsonID.contains("index"))
    idx = jsonID.at("index").get<uint32_t>();
  return AppID(jsonID.at("name").get<string>(), idx);
}

static AppIDPath parseIDPath(const nlohmann::json &jsonIDPath) {
  AppIDPath ret;
  for (auto &id : jsonIDPath)
    ret.push_back(parseID(id));
  return ret;
}

static ServicePortDesc parseServicePort(const nlohmann::json &jsonPort) {
  return ServicePortDesc{jsonPort.at("outer_sym").get<string>(),
                         jsonPort.at("inner").get<string>()};
}

/// Convert the json value to a 'any', which can be exposed outside of this
/// file.
static any getAny(const nlohmann::json &value) {
  auto getObject = [](const nlohmann::json &json) {
    map<string, any> ret;
    for (auto &e : json.items())
      ret[e.key()] = getAny(e.value());
    return ret;
  };

  auto getArray = [](const nlohmann::json &json) {
    vector<any> ret;
    for (auto &e : json)
      ret.push_back(getAny(e));
    return ret;
  };

  if (value.is_string())
    return value.get<string>();
  else if (value.is_number_integer())
    return value.get<int64_t>();
  else if (value.is_number_unsigned())
    return value.get<uint64_t>();
  else if (value.is_number_float())
    return value.get<double>();
  else if (value.is_boolean())
    return value.get<bool>();
  else if (value.is_null())
    return value.get<nullptr_t>();
  else if (value.is_object())
    return getObject(value);
  else if (value.is_array())
    return getArray(value);
  else
    throw runtime_error("Unknown type in manifest: " + value.dump(2));
}

static ModuleInfo parseModuleInfo(const nlohmann::json &mod) {

  map<string, any> extras;
  for (auto &extra : mod.items())
    if (extra.key() != "name" && extra.key() != "summary" &&
        extra.key() != "version" && extra.key() != "repo" &&
        extra.key() != "commitHash" && extra.key() != "symbolRef")
      extras[extra.key()] = getAny(extra.value());

  auto value = [&](const string &key) -> optional<string> {
    auto f = mod.find(key);
    if (f == mod.end())
      return nullopt;
    return f.value();
  };
  return ModuleInfo{value("name"), value("summary"),    value("version"),
                    value("repo"), value("commitHash"), extras};
}

//===----------------------------------------------------------------------===//
// Manifest::Impl class implementation.
//===----------------------------------------------------------------------===//

Manifest::Impl::Impl(Context &ctxt, const string &manifestStr) : ctxt(ctxt) {
  manifestJson = nlohmann::ordered_json::parse(manifestStr);

  for (auto &mod : manifestJson.at("symbols"))
    symbolInfoCache.insert(
        make_pair(mod.at("symbolRef"), parseModuleInfo(mod)));
  populateTypes(manifestJson.at("types"));
}

unique_ptr<Accelerator>
Manifest::Impl::buildAccelerator(AcceleratorConnection &acc) const {
  ServiceTable activeSvcs;

  // Get the initial active services table. Update it as we descend down.
  auto svcDecls = manifestJson.at("service_decls");
  scanServiceDecls(acc, svcDecls, activeSvcs);

  // Get the services instantiated at the top level.
  auto designJson = manifestJson.at("design");
  vector<services::Service *> services =
      getServices({}, acc, designJson, activeSvcs);

  // Get the ports at the top level.
  auto ports = getBundlePorts(acc, {}, activeSvcs, designJson);

  return make_unique<Accelerator>(
      getModInfo(designJson),
      getChildInstances({}, acc, activeSvcs, designJson), services, ports);
}

optional<ModuleInfo>
Manifest::Impl::getModInfo(const nlohmann::json &json) const {
  auto instOfIter = json.find("inst_of");
  if (instOfIter == json.end())
    return nullopt;
  auto f = symbolInfoCache.find(instOfIter.value());
  if (f != symbolInfoCache.end())
    return f->second;
  return nullopt;
}

void Manifest::Impl::scanServiceDecls(AcceleratorConnection &acc,
                                      const nlohmann::json &svcDecls,
                                      ServiceTable &activeServices) const {
  for (auto &svcDecl : svcDecls) {
    if (auto f = svcDecl.find("type_name"); f != svcDecl.end()) {
      // Get the implementation details.
      ServiceImplDetails svcDetails;
      for (auto &detail : svcDecl.items())
        svcDetails[detail.key()] = getAny(detail.value());

      // Create the service.
      services::Service::Type svcId =
          services::ServiceRegistry::lookupServiceType(f.value());
      auto svc = acc.getService(svcId, /*id=*/{}, /*implName=*/"",
                                /*details=*/svcDetails, /*clients=*/{});
      if (svc)
        activeServices[svcDecl.at("symbol")] = svc;
    }
  }
}

vector<unique_ptr<Instance>>
Manifest::Impl::getChildInstances(AppIDPath idPath, AcceleratorConnection &acc,
                                  const ServiceTable &activeServices,
                                  const nlohmann::json &instJson) const {
  vector<unique_ptr<Instance>> ret;
  auto childrenIter = instJson.find("children");
  if (childrenIter == instJson.end())
    return ret;
  for (auto &child : childrenIter.value())
    ret.emplace_back(getChildInstance(idPath, acc, activeServices, child));
  return ret;
}

unique_ptr<Instance>
Manifest::Impl::getChildInstance(AppIDPath idPath, AcceleratorConnection &acc,
                                 ServiceTable activeServices,
                                 const nlohmann::json &child) const {
  AppID childID = parseID(child.at("app_id"));
  idPath.push_back(childID);

  vector<services::Service *> services =
      getServices(idPath, acc, child, activeServices);

  auto children = getChildInstances(idPath, acc, activeServices, child);
  auto ports = getBundlePorts(acc, idPath, activeServices, child);
  return make_unique<Instance>(parseID(child.at("app_id")), getModInfo(child),
                               std::move(children), services, ports);
}

services::Service *
Manifest::Impl::getService(AppIDPath idPath, AcceleratorConnection &acc,
                           const nlohmann::json &svcJson,
                           ServiceTable &activeServices) const {

  AppID id = parseID(svcJson.at("appID"));
  idPath.push_back(id);

  // Get all the client info, including the implementation details.
  HWClientDetails clientDetails;
  for (auto &client : svcJson.at("client_details")) {
    HWClientDetail clientDetail;
    for (auto &detail : client.items()) {
      if (detail.key() == "relAppIDPath")
        clientDetail.relPath = parseIDPath(detail.value());
      else if (detail.key() == "port")
        clientDetail.port = parseServicePort(detail.value());
      else
        clientDetail.implOptions[detail.key()] = getAny(detail.value());
    }
    clientDetails.push_back(clientDetail);
  }

  // Get the implementation details.
  ServiceImplDetails svcDetails;
  std::string implName;
  std::string service;
  for (auto &detail : svcJson.items()) {
    if (detail.key() == "appID" || detail.key() == "client_details")
      continue;
    if (detail.key() == "serviceImplName")
      implName = detail.value();
    else if (detail.key() == "service")
      service = detail.value().get<std::string>().substr(1);
    else
      svcDetails[detail.key()] = getAny(detail.value());
  }

  // Create the service.
  // TODO: Add support for 'standard' services.
  services::Service::Type svcType =
      services::ServiceRegistry::lookupServiceType(service);
  services::Service *svc =
      acc.getService(svcType, idPath, implName, svcDetails, clientDetails);
  if (svc)
    // Update the active services table.
    activeServices[service] = svc;
  return svc;
}

vector<services::Service *>
Manifest::Impl::getServices(AppIDPath idPath, AcceleratorConnection &acc,
                            const nlohmann::json &svcsJson,
                            ServiceTable &activeServices) const {
  vector<services::Service *> ret;
  auto contentsIter = svcsJson.find("contents");
  if (contentsIter == svcsJson.end())
    return ret;

  for (auto &content : contentsIter.value())
    if (content.at("class") == "service")
      ret.emplace_back(getService(idPath, acc, content, activeServices));
  return ret;
}

vector<std::unique_ptr<BundlePort>>
Manifest::Impl::getBundlePorts(AcceleratorConnection &acc, AppIDPath idPath,
                               const ServiceTable &activeServices,
                               const nlohmann::json &instJson) const {
  vector<std::unique_ptr<BundlePort>> ret;
  auto contentsIter = instJson.find("contents");
  if (contentsIter == instJson.end())
    return ret;

  for (auto &content : contentsIter.value()) {
    if (content.at("class") != "client_port")
      continue;

    // Lookup the requested service in the active services table.
    std::string serviceName = "";
    if (auto f = content.find("servicePort"); f != content.end())
      serviceName = parseServicePort(f.value()).name;
    auto svcIter = activeServices.find(serviceName);
    if (svcIter == activeServices.end()) {
      // If a specific service isn't found, search for the default service
      // (typically provided by a BSP).
      if (svcIter = activeServices.find(""); svcIter == activeServices.end())
        throw runtime_error(
            "Malformed manifest: could not find active service '" +
            serviceName + "'");
    }
    services::Service *svc = svcIter->second;

    string typeName = content.at("bundleType").at("circt_name");
    auto type = getType(typeName);
    if (!type)
      throw runtime_error("Malformed manifest: could not find port type '" +
                          typeName + "'");
    const BundleType *bundleType = dynamic_cast<const BundleType *>(*type);
    if (!bundleType)
      throw runtime_error("Malformed manifest: type '" + typeName +
                          "' is not a bundle type");

    idPath.push_back(parseID(content.at("appID")));
    map<string, ChannelPort &> portChannels =
        acc.requestChannelsFor(idPath, bundleType);

    services::ServicePort *svcPort =
        svc->getPort(idPath, bundleType, portChannels, acc);
    if (svcPort)
      ret.emplace_back(svcPort);
    else
      ret.emplace_back(new BundlePort(idPath.back(), portChannels));
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

  vector<tuple<string, BundleType::Direction, const Type *>> channels;
  for (auto &chanJson : typeJson["channels"]) {
    string dirStr = chanJson.at("direction");
    BundleType::Direction dir;
    if (dirStr == "to")
      dir = BundleType::Direction::To;
    else if (dirStr == "from")
      dir = BundleType::Direction::From;
    else
      throw runtime_error("Malformed manifest: unknown direction '" + dirStr +
                          "'");
    channels.emplace_back(chanJson.at("name"), dir,
                          parseType(chanJson["type"], cache));
  }
  return new BundleType(typeJson.at("circt_name"), channels);
}

ChannelType *parseChannelType(const nlohmann::json &typeJson, Context &cache) {
  assert(typeJson.at("mnemonic") == "channel");
  return new ChannelType(typeJson.at("circt_name"),
                         parseType(typeJson.at("inner"), cache));
}

Type *parseInt(const nlohmann::json &typeJson, Context &cache) {
  assert(typeJson.at("mnemonic") == "int");
  std::string sign = typeJson.at("signedness");
  uint64_t width = typeJson.at("hw_bitwidth");
  Type::ID id = typeJson.at("circt_name");

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
    throw runtime_error("Malformed manifest: unknown sign '" + sign + "'");
}

StructType *parseStruct(const nlohmann::json &typeJson, Context &cache) {
  assert(typeJson.at("mnemonic") == "struct");
  vector<pair<string, const Type *>> fields;
  for (auto &fieldJson : typeJson["fields"])
    fields.emplace_back(fieldJson.at("name"),
                        parseType(fieldJson["type"], cache));
  return new StructType(typeJson.at("circt_name"), fields);
}

ArrayType *parseArray(const nlohmann::json &typeJson, Context &cache) {
  assert(typeJson.at("mnemonic") == "array");
  uint64_t size = typeJson.at("size");
  return new ArrayType(typeJson.at("circt_name"),
                       parseType(typeJson.at("element"), cache), size);
}

using TypeParser = std::function<Type *(const nlohmann::json &, Context &)>;
const std::map<std::string_view, TypeParser> typeParsers = {
    {"bundle", parseBundleType},
    {"channel", parseChannelType},
    {"any",
     [](const nlohmann::json &typeJson, Context &cache) {
       return new AnyType(typeJson.at("circt_name"));
     }},
    {"int", parseInt},
    {"struct", parseStruct},
    {"array", parseArray},

};

// Parse a type if it doesn't already exist in the cache.
const Type *parseType(const nlohmann::json &typeJson, Context &cache) {
  // We use the circt type string as a unique ID.
  string circt_name = typeJson.at("circt_name");
  if (optional<const Type *> t = cache.getType(circt_name))
    return *t;

  string mnemonic = typeJson.at("mnemonic");
  Type *t;
  auto f = typeParsers.find(mnemonic);
  if (f != typeParsers.end())
    t = f->second(typeJson, cache);
  else
    // Types we don't know about are opaque.
    t = new Type(circt_name);

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

Manifest::Manifest(Context &ctxt, const string &jsonManifest)
    : impl(new Impl(ctxt, jsonManifest)) {}

Manifest::~Manifest() { delete impl; }

uint32_t Manifest::getApiVersion() const {
  return impl->at("api_version").get<uint32_t>();
}

vector<ModuleInfo> Manifest::getModuleInfos() const {
  vector<ModuleInfo> ret;
  for (auto &mod : impl->at("symbols"))
    ret.push_back(parseModuleInfo(mod));
  return ret;
}

unique_ptr<Accelerator>
Manifest::buildAccelerator(AcceleratorConnection &acc) const {
  return impl->buildAccelerator(acc);
}

const vector<const Type *> &Manifest::getTypeTable() const {
  return impl->getTypeTable();
}

//===----------------------------------------------------------------------===//
// POCO helpers.
//===----------------------------------------------------------------------===//

// Print a module info, including the extra metadata.
ostream &operator<<(ostream &os, const ModuleInfo &m) {
  auto printAny = [&os](any a) {
    const type_info &t = a.type();
    if (t == typeid(string))
      os << any_cast<string>(a);
    else if (t == typeid(int64_t))
      os << any_cast<int64_t>(a);
    else if (t == typeid(uint64_t))
      os << any_cast<uint64_t>(a);
    else if (t == typeid(double))
      os << any_cast<double>(a);
    else if (t == typeid(bool))
      os << any_cast<bool>(a);
    else if (t == typeid(nullptr_t))
      os << "null";
    else
      os << "unknown";
  };

  if (m.name)
    os << *m.name << " ";
  if (m.version)
    os << *m.version << " ";
  if (m.repo || m.commitHash) {
    os << "(";
    if (m.repo)
      os << *m.repo;
    if (m.commitHash)
      os << "@" << *m.commitHash;
    os << ")";
  }
  if (m.summary)
    os << ": " << *m.summary;
  os << "\n";

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
AppIDPath AppIDPath::operator+(const AppIDPath &b) {
  AppIDPath ret = *this;
  ret.insert(ret.end(), b.begin(), b.end());
  return ret;
}

string AppIDPath::toStr() const {
  ostringstream os;
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
} // namespace esi

ostream &operator<<(ostream &os, const AppID &id) {
  os << id.name;
  if (id.idx)
    os << "[" << *id.idx << "]";
  return os;
}
ostream &operator<<(ostream &os, const AppIDPath &path) {
  for (size_t i = 0, e = path.size(); i < e; ++i) {
    if (i > 0)
      os << '.';
    os << path[i];
  }
  return os;
}
