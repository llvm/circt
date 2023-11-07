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
// should always be modified within CIRCT
// (lib/dialect/ESI/runtime/cpp/lib/backends/Cosim.cpp).
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

// This is a proxy class to the manifest JSON. It is used to avoid having to
// include the JSON parser in the header. Forward references don't work since
// nlohmann::json is a rather complex template.
//
// Plus, it allows us to hide some implementation functions from the header
// file.
class ManifestProxy {
public:
  ManifestProxy(std::string jsonManifest, Manifest &manifest);

  auto at(std::string key) const { return manifestJson.at(key); }
  std::optional<ModuleInfo> getModInfo(const nlohmann::json &json) const;
  Instance getInstance(const nlohmann::json &json) const;
  std::vector<std::unique_ptr<Instance>>
  getChildInstances(Accelerator &acc, const nlohmann::json &json) const;
  std::vector<std::unique_ptr<services::Service>>
  getServiceProviders(Accelerator &acc, const nlohmann::json &contents) const;

  std::unique_ptr<Design> buildDesign(Accelerator &acc) const;

private:
  nlohmann::json manifestJson;
  Manifest &manifest;
  std::map<std::string, ModuleInfo> symbolInfoCache;
};
} // namespace internal
} // namespace esi

//===----------------------------------------------------------------------===//
// Simple JSON -> object parsers.
//===----------------------------------------------------------------------===//

static AppID parseID(const nlohmann::json &json) {
  std::optional<uint32_t> idx;
  if (json.contains("index"))
    idx = json.at("index").get<uint32_t>();
  return AppID{json.at("name").get<std::string>(), idx};
}

static ModuleInfo parseModuleInfo(const nlohmann::json &mod) {
  auto getAny = [](const nlohmann::json &json) -> std::any {
    if (json.is_string())
      return json.get<std::string>();
    else if (json.is_number_integer())
      return json.get<int64_t>();
    else if (json.is_number_unsigned())
      return json.get<uint64_t>();
    else if (json.is_number_float())
      return json.get<double>();
    else if (json.is_boolean())
      return json.get<bool>();
    else if (json.is_null())
      return json.get<std::nullptr_t>();
    else
      throw std::runtime_error("Unknown type in manifest: " + json.dump(2));
  };
  std::map<std::string, std::any> extras;
  for (auto &extra : mod.items())
    if (extra.key() != "name" && extra.key() != "summary" &&
        extra.key() != "version" && extra.key() != "repo" &&
        extra.key() != "commit_hash" && extra.key() != "symbolRef")
      extras[extra.key()] = getAny(extra.value());

  auto value = [&](std::string key) -> std::optional<std::string> {
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

internal::ManifestProxy::ManifestProxy(std::string manifestStr,
                                       Manifest &manifest)
    : manifest(manifest) {
  manifestJson = nlohmann::ordered_json::parse(manifestStr);

  for (auto &mod : manifestJson.at("symbols"))
    symbolInfoCache.insert(
        std::make_pair(mod.at("symbolRef"), parseModuleInfo(mod)));
}

std::unique_ptr<Design>
internal::ManifestProxy::buildDesign(Accelerator &acc) const {
  auto designJson = manifestJson.at("design");
  std::vector<std::unique_ptr<services::Service>> services =
      getServiceProviders(acc, designJson);
  std::vector<std::unique_ptr<Instance>> children =
      getChildInstances(acc, designJson);
  return std::make_unique<Design>(getModInfo(designJson), std::move(children),
                                  std::move(services));
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
internal::ManifestProxy::getChildInstances(Accelerator &acc,
                                           const nlohmann::json &json) const {
  std::vector<std::unique_ptr<Instance>> ret;
  auto childrenIter = json.find("children");
  if (childrenIter == json.end())
    return ret;
  for (auto &child : childrenIter.value()) {
    std::vector<std::unique_ptr<services::Service>> services =
        getServiceProviders(acc, child);
    auto children = getChildInstances(acc, child);
    ret.emplace_back(std::make_unique<Instance>(
        parseID(child.at("app_id")), getModInfo(child), std::move(children),
        std::move(services)));
  }
  return ret;
}
std::vector<std::unique_ptr<services::Service>>
internal::ManifestProxy::getServiceProviders(
    Accelerator &acc, const nlohmann::json &contents) const {
  std::vector<std::unique_ptr<services::Service>> ret;
  return ret;
}

//===----------------------------------------------------------------------===//
// Manifest class implementation.
//===----------------------------------------------------------------------===//

Manifest::Manifest(std::string jsonManifest)
    : manifest(*new internal::ManifestProxy(jsonManifest, *this)) {}
Manifest::~Manifest() { delete &manifest; }

uint32_t Manifest::apiVersion() const {
  return manifest.at("api_version").get<uint32_t>();
}

std::vector<ModuleInfo> Manifest::modules() const {
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

std::ostream &operator<<(std::ostream &os, const ModuleInfo &m) {
  auto printAny = [&os](std::any a) {
    if (a.type() == typeid(std::string))
      os << std::any_cast<std::string>(a);
    else if (a.type() == typeid(int64_t))
      os << std::any_cast<int64_t>(a);
    else if (a.type() == typeid(uint64_t))
      os << std::any_cast<uint64_t>(a);
    else if (a.type() == typeid(double))
      os << std::any_cast<double>(a);
    else if (a.type() == typeid(bool))
      os << std::any_cast<bool>(a);
    else if (a.type() == typeid(std::nullptr_t))
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
