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

using namespace esi;

namespace esi {
namespace internal {

// While building the design, keep around a map of active services indexed by
// the service name.
using ServiceTable = std::map<std::string, services::Service *>;

// This is a proxy class to the manifest JSON. It is used to avoid having to
// include the JSON parser in the header. Forward references don't work since
// nlohmann::json is a rather complex template.
//
// Plus, it allows us to hide some implementation functions from the header
// file.
class ManifestProxy {
public:
  ManifestProxy(const std::string &jsonManifest, Manifest &manifest);

  auto at(const std::string &key) const { return manifestJson.at(key); }

  // Get the module info (if any) for the module instance in 'json'.
  std::optional<ModuleInfo> getModInfo(const nlohmann::json &) const;

  services::Service *getService(AppIDPath idPath, Accelerator &,
                                const nlohmann::json &,
                                ServiceTable &activeServices) const;
  void getServices(AppIDPath idPath, Accelerator &, const nlohmann::json &,
                   std::vector<services::Service *> &outServices,
                   ServiceTable &activeServices) const;

  std::vector<BundlePort> getBundlePorts(AppIDPath idPath,
                                         const ServiceTable &activeServices,
                                         const nlohmann::json &json) const;

  /// Build the set of child instances (recursively) for the module instance in
  /// 'json'.
  std::vector<std::unique_ptr<Instance>>
  getChildInstances(AppIDPath idPath, Accelerator &acc,
                    ServiceTable activeServices,
                    const nlohmann::json &json) const;
  std::unique_ptr<Instance> getChildInstance(AppIDPath idPath, Accelerator &acc,
                                             ServiceTable activeServices,
                                             const nlohmann::json &json) const;
  /// Build a dynamic API for the Accelerator connection 'acc'
  /// based on the manifest stored herein.
  std::unique_ptr<Design> buildDesign(Accelerator &acc) const;

private:
  // The parsed json.
  nlohmann::json manifestJson;
  // Cache the module info for each symbol.
  std::map<std::string, ModuleInfo> symbolInfoCache;
};
} // namespace internal
} // namespace esi

//===----------------------------------------------------------------------===//
// Simple JSON -> object parsers.
//===----------------------------------------------------------------------===//

static AppID parseID(const nlohmann::json &jsonID) {
  std::optional<uint32_t> idx;
  if (jsonID.contains("index"))
    idx = jsonID.at("index").get<uint32_t>();
  return AppID{jsonID.at("name").get<std::string>(), idx};
}

static AppIDPath parseIDPath(const nlohmann::json &jsonIDPath) {
  AppIDPath ret;
  for (auto &id : jsonIDPath)
    ret.push_back(parseID(id));
  return ret;
}

static services::ServicePortDesc
parseServicePort(const nlohmann::json &jsonPort) {
  return services::ServicePortDesc{jsonPort.at("outer_sym").get<std::string>(),
                                   jsonPort.at("inner").get<std::string>()};
}

static std::any getAny(const nlohmann::json &value) {
  auto getObject = [](const nlohmann::json &json) {
    std::map<std::string, std::any> ret;
    for (auto &e : json.items())
      ret[e.key()] = getAny(e.value());
    return ret;
  };

  auto getArray = [](const nlohmann::json &json) {
    std::vector<std::any> ret;
    for (auto &e : json)
      ret.push_back(getAny(e));
    return ret;
  };

  if (value.is_string())
    return value.get<std::string>();
  else if (value.is_number_integer())
    return value.get<int64_t>();
  else if (value.is_number_unsigned())
    return value.get<uint64_t>();
  else if (value.is_number_float())
    return value.get<double>();
  else if (value.is_boolean())
    return value.get<bool>();
  else if (value.is_null())
    return value.get<std::nullptr_t>();
  else if (value.is_object())
    return getObject(value);
  else if (value.is_array())
    return getArray(value);
  else
    throw std::runtime_error("Unknown type in manifest: " + value.dump(2));
}

static ModuleInfo parseModuleInfo(const nlohmann::json &mod) {

  std::map<std::string, std::any> extras;
  for (auto &extra : mod.items())
    if (extra.key() != "name" && extra.key() != "summary" &&
        extra.key() != "version" && extra.key() != "repo" &&
        extra.key() != "commit_hash" && extra.key() != "symbolRef")
      extras[extra.key()] = getAny(extra.value());

  auto value = [&](const std::string &key) -> std::optional<std::string> {
    auto f = mod.find(key);
    if (f == mod.end())
      return std::nullopt;
    return f.value();
  };
  return ModuleInfo{value("name"), value("summary"),     value("version"),
                    value("repo"), value("commit_hash"), extras};
}

//===----------------------------------------------------------------------===//
// ManifestProxy class implementation.
//===----------------------------------------------------------------------===//

internal::ManifestProxy::ManifestProxy(const std::string &manifestStr,
                                       Manifest &manifest) {
  manifestJson = nlohmann::ordered_json::parse(manifestStr);

  for (auto &mod : manifestJson.at("symbols"))
    symbolInfoCache.insert(
        std::make_pair(mod.at("symbolRef"), parseModuleInfo(mod)));
}

std::unique_ptr<Design>
internal::ManifestProxy::buildDesign(Accelerator &acc) const {
  auto designJson = manifestJson.at("design");

  ServiceTable activeSvcs;
  std::vector<services::Service *> services;
  getServices({}, acc, designJson, services, activeSvcs);

  return std::make_unique<Design>(
      getModInfo(designJson),
      getChildInstances({}, acc, activeSvcs, designJson), services,
      getBundlePorts({}, activeSvcs, designJson));
}

std::optional<ModuleInfo>
internal::ManifestProxy::getModInfo(const nlohmann::json &json) const {
  auto instOfIter = json.find("inst_of");
  if (instOfIter == json.end())
    return std::nullopt;
  auto f = symbolInfoCache.find(instOfIter.value());
  if (f != symbolInfoCache.end())
    return f->second;
  return std::nullopt;
}

std::vector<std::unique_ptr<Instance>>
internal::ManifestProxy::getChildInstances(
    AppIDPath idPath, Accelerator &acc, ServiceTable activeServices,
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
internal::ManifestProxy::getChildInstance(AppIDPath idPath, Accelerator &acc,
                                          ServiceTable activeServices,
                                          const nlohmann::json &child) const {
  AppID childID = parseID(child.at("app_id"));
  idPath.push_back(childID);

  std::vector<services::Service *> services;
  getServices(idPath, acc, child, services, activeServices);

  auto children = getChildInstances(idPath, acc, activeServices, child);
  return std::make_unique<Instance>(
      parseID(child.at("app_id")), getModInfo(child), std::move(children),
      services, getBundlePorts(idPath, activeServices, child));
}

services::Service *
internal::ManifestProxy::getService(AppIDPath idPath, Accelerator &acc,
                                    const nlohmann::json &svcJson,
                                    ServiceTable &activeServices) const {

  AppID id = parseID(svcJson.at("appID"));
  idPath.push_back(id);

  services::HWClientDetails clientDetails;
  for (auto &client : svcJson.at("client_details")) {
    services::HWClientDetail clientDetail;
    for (auto &detail : client.items()) {
      if (detail.key() == "relAppIDPath")
        clientDetail.path = parseIDPath(detail.value());
      else if (detail.key() == "port")
        clientDetail.port = parseServicePort(detail.value());
      else
        clientDetail.implOptions[detail.key()] = getAny(detail.value());
    }
  }

  services::ServiceImplDetails svcDetails;
  for (auto &detail : svcJson.items())
    if (detail.key() != "appID" && detail.key() != "client_details")
      svcDetails[detail.key()] = getAny(detail.value());

  auto svc = acc.getService<services::CustomService>(idPath, svcDetails,
                                                     clientDetails);
  if (!svc) {
    throw std::runtime_error("Could not create service for ");
  }
  activeServices[svc->getServiceSymbol()] = svc;
  return svc;
}

void internal::ManifestProxy::getServices(AppIDPath idPath, Accelerator &acc,
                                          const nlohmann::json &svcsJson,
                                          std::vector<services::Service *> &ret,
                                          ServiceTable &activeServices) const {

  auto contentsIter = svcsJson.find("contents");
  if (contentsIter == svcsJson.end())
    return;

  for (auto &content : contentsIter.value())
    if (content.at("class") == "service")
      ret.emplace_back(getService(idPath, acc, content, activeServices));
}

std::vector<BundlePort>
internal::ManifestProxy::getBundlePorts(AppIDPath idPath,
                                        const ServiceTable &activeServices,
                                        const nlohmann::json &instJson) const {

  std::vector<BundlePort> ret;
  auto contentsIter = instJson.find("contents");
  if (contentsIter == instJson.end())
    return ret;

  for (auto &content : contentsIter.value()) {
    if (content.at("class") != "client_port")
      continue;

    services::ServicePortDesc port =
        parseServicePort(content.at("servicePort"));
    auto f = activeServices.find(port.name);
    if (f == activeServices.end())
      throw std::runtime_error("Could not find service '" + port.name + "'");

    idPath.push_back(parseID(content.at("appID")));
    ret.emplace_back(idPath.back(), f->second->requestChannelsFor(idPath));
    idPath.pop_back();
  }
  return ret;
}

//===----------------------------------------------------------------------===//
// Manifest class implementation.
//===----------------------------------------------------------------------===//

Manifest::Manifest(const std::string &jsonManifest)
    : manifest(*new internal::ManifestProxy(jsonManifest, *this)) {}
Manifest::~Manifest() { delete &manifest; }

uint32_t Manifest::apiVersion() const {
  return manifest.at("api_version").get<uint32_t>();
}

std::vector<ModuleInfo> Manifest::moduleInfos() const {
  std::vector<ModuleInfo> ret;
  for (auto &mod : manifest.at("symbols"))
    ret.push_back(parseModuleInfo(mod));
  return ret;
}

std::unique_ptr<Design> Manifest::buildDesign(Accelerator &acc) const {
  return manifest.buildDesign(acc);
}

//===----------------------------------------------------------------------===//
// POCO helpers.
//===----------------------------------------------------------------------===//

// Print a module info, including the extra metadata.
std::ostream &operator<<(std::ostream &os, const ModuleInfo &m) {
  auto printAny = [&os](std::any a) {
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
