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
#include "esi/Design.h"
#include "esi/StdServices.h"

#include <nlohmann/json.hpp>

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
  Impl(const string &jsonManifest);

  auto at(const string &key) const { return manifestJson.at(key); }

  // Get the module info (if any) for the module instance in 'json'.
  optional<ModuleInfo> getModInfo(const nlohmann::json &) const;

  /// Get a Service for the service specified in 'json'. Update the
  /// activeServices table.
  services::Service *getService(AppIDPath idPath, Accelerator &,
                                const nlohmann::json &,
                                ServiceTable &activeServices) const;

  /// Get all the services in the description of an instance. Update the active
  /// services table.
  vector<services::Service *> getServices(AppIDPath idPath, Accelerator &,
                                          const nlohmann::json &,
                                          ServiceTable &activeServices) const;

  /// Get the bundle ports for the instance at 'idPath' and specified in
  /// 'instJson'. Look them up in 'activeServies'.
  vector<BundlePort> getBundlePorts(AppIDPath idPath,
                                    const ServiceTable &activeServices,
                                    const nlohmann::json &instJson) const;

  /// Build the set of child instances (recursively) for the module instance
  /// description.
  vector<unique_ptr<Instance>>
  getChildInstances(AppIDPath idPath, Accelerator &acc,
                    const ServiceTable &activeServices,
                    const nlohmann::json &instJson) const;

  /// Get a single child instance. Implicitly copy the active services table so
  /// that it can be safely updated for the child's branch of the tree.
  unique_ptr<Instance> getChildInstance(AppIDPath idPath, Accelerator &acc,
                                        ServiceTable activeServices,
                                        const nlohmann::json &childJson) const;

  /// Parse all the types and populate the types table.
  void populateTypes(const nlohmann::json &typesJson);

  // Forwarded from Manifest.
  const vector<reference_wrapper<const Type>> &getTypeTable() const {
    return _typeTable;
  }

  // Forwarded from Manifest.
  optional<reference_wrapper<const Type>> getType(Type::ID id) const {
    if (auto f = _types.find(id); f != _types.end())
      return *f->second;
    return nullopt;
  }

  /// Build a dynamic API for the Accelerator connection 'acc' based on the
  /// manifest stored herein.
  unique_ptr<Design> buildDesign(Accelerator &acc) const;

  const Type &parseType(const nlohmann::json &typeJson);

private:
  BundleType *parseBundleType(const nlohmann::json &typeJson);

  vector<reference_wrapper<const Type>> _typeTable;
  map<Type::ID, unique_ptr<Type>> _types;

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
        extra.key() != "commit_hash" && extra.key() != "symbolRef")
      extras[extra.key()] = getAny(extra.value());

  auto value = [&](const string &key) -> optional<string> {
    auto f = mod.find(key);
    if (f == mod.end())
      return nullopt;
    return f.value();
  };
  return ModuleInfo{value("name"), value("summary"),     value("version"),
                    value("repo"), value("commit_hash"), extras};
}

//===----------------------------------------------------------------------===//
// ManifestProxy class implementation.
//===----------------------------------------------------------------------===//

Manifest::Impl::Impl(const string &manifestStr) {
  manifestJson = nlohmann::ordered_json::parse(manifestStr);

  for (auto &mod : manifestJson.at("symbols"))
    symbolInfoCache.insert(
        make_pair(mod.at("symbolRef"), parseModuleInfo(mod)));
  populateTypes(manifestJson.at("types"));
}

unique_ptr<Design> Manifest::Impl::buildDesign(Accelerator &acc) const {
  auto designJson = manifestJson.at("design");

  // Get the initial active services table. Update it as we descend down.
  ServiceTable activeSvcs;
  vector<services::Service *> services =
      getServices({}, acc, designJson, activeSvcs);

  return make_unique<Design>(getModInfo(designJson),
                             getChildInstances({}, acc, activeSvcs, designJson),
                             services,
                             getBundlePorts({}, activeSvcs, designJson));
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

vector<unique_ptr<Instance>>
Manifest::Impl::getChildInstances(AppIDPath idPath, Accelerator &acc,
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
Manifest::Impl::getChildInstance(AppIDPath idPath, Accelerator &acc,
                                 ServiceTable activeServices,
                                 const nlohmann::json &child) const {
  AppID childID = parseID(child.at("app_id"));
  idPath.push_back(childID);

  vector<services::Service *> services =
      getServices(idPath, acc, child, activeServices);

  auto children = getChildInstances(idPath, acc, activeServices, child);
  return make_unique<Instance>(parseID(child.at("app_id")), getModInfo(child),
                               move(children), services,
                               getBundlePorts(idPath, activeServices, child));
}

services::Service *
Manifest::Impl::getService(AppIDPath idPath, Accelerator &acc,
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
  for (auto &detail : svcJson.items())
    if (detail.key() != "appID" && detail.key() != "client_details")
      svcDetails[detail.key()] = getAny(detail.value());

  // Create the service.
  // TODO: Add support for 'standard' services.
  auto svc = acc.getService<services::CustomService>(idPath, svcDetails,
                                                     clientDetails);
  if (!svc)
    throw runtime_error("Could not create service for ");

  // Update the active services table.
  activeServices[svc->getServiceSymbol()] = svc;
  return svc;
}

vector<services::Service *>
Manifest::Impl::getServices(AppIDPath idPath, Accelerator &acc,
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

vector<BundlePort>
Manifest::Impl::getBundlePorts(AppIDPath idPath,
                               const ServiceTable &activeServices,
                               const nlohmann::json &instJson) const {
  vector<BundlePort> ret;
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
    auto svc = activeServices.find(serviceName);
    if (svc == activeServices.end())
      throw runtime_error(
          "Malformed manifest: could not find active service '" + serviceName +
          "'");

    string typeName = content.at("bundleType").at("circt_name");
    auto type = getType(typeName);
    if (!type)
      throw runtime_error("Malformed manifest: could not find port type '" +
                          typeName + "'");
    const BundleType &bundleType =
        dynamic_cast<const BundleType &>(type->get());

    BundlePort::Direction portDir;
    string dirStr = content.at("direction");
    if (dirStr == "toClient")
      portDir = BundlePort::Direction::ToClient;
    else if (dirStr == "toServer")
      portDir = BundlePort::Direction::ToServer;
    else
      throw runtime_error("Malformed manifest: unknown direction '" + dirStr +
                          "'");

    idPath.push_back(parseID(content.at("appID")));
    map<string, ChannelPort &> portChannels;
    // If we need to have custom ports (because of a custom service), add them.
    if (auto *customSvc = dynamic_cast<services::CustomService *>(svc->second))
      portChannels = customSvc->requestChannelsFor(idPath, bundleType, portDir);
    ret.emplace_back(idPath.back(), portChannels);
    // Since we share idPath between iterations, pop the last element before the
    // next iteration.
    idPath.pop_back();
  }
  return ret;
}

BundleType *Manifest::Impl::parseBundleType(const nlohmann::json &typeJson) {
  assert(typeJson.at("mnemonic") == "bundle");

  vector<tuple<string, BundleType::Direction, const Type &>> channels;
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
                          parseType(chanJson["type"]));
  }
  return new BundleType(typeJson.at("circt_name"), channels);
}

// Parse a type if it doesn't already exist in the cache.
const Type &Manifest::Impl::parseType(const nlohmann::json &typeJson) {
  // We use the circt type string as a unique ID.
  string circt_name = typeJson.at("circt_name");

  // Check the cache.
  auto typeF = _types.find(circt_name);
  if (typeF != _types.end())
    return *typeF->second;

  // Parse the type.
  string mnemonic = typeJson.at("mnemonic");
  Type *t;
  if (mnemonic == "bundle")
    t = parseBundleType(typeJson);
  else
    // Types we don't know about are opaque.
    t = new Type(circt_name);

  // Insert into the cache.
  _types.emplace(circt_name, unique_ptr<Type>(t));
  return *t;
}

void Manifest::Impl::populateTypes(const nlohmann::json &typesJson) {
  for (auto &typeJson : typesJson)
    _typeTable.push_back(parseType(typeJson));
}

//===----------------------------------------------------------------------===//
// Manifest class implementation.
//===----------------------------------------------------------------------===//

Manifest::Manifest(const string &jsonManifest)
    : impl(*new Impl(jsonManifest)) {}
Manifest::~Manifest() { delete &impl; }

uint32_t Manifest::getApiVersion() const {
  return impl.at("api_version").get<uint32_t>();
}

vector<ModuleInfo> Manifest::getModuleInfos() const {
  vector<ModuleInfo> ret;
  for (auto &mod : impl.at("symbols"))
    ret.push_back(parseModuleInfo(mod));
  return ret;
}

unique_ptr<Design> Manifest::buildDesign(Accelerator &acc) const {
  return impl.buildDesign(acc);
}

optional<reference_wrapper<const Type>> Manifest::getType(Type::ID id) const {
  if (auto f = impl._types.find(id); f != impl._types.end())
    return *f->second;
  return nullopt;
}

const vector<reference_wrapper<const Type>> &Manifest::getTypeTable() const {
  return impl.getTypeTable();
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
    os << ": " << *m.summary << "\n";

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
} // namespace esi
