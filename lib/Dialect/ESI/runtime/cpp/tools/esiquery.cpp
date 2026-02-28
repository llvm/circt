//===- esiquery.cpp - ESI accelerator system query tool -------------------===//
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

#include "esi/Accelerator.h"
#include "esi/CLI.h"
#include "esi/Manifest.h"
#include "esi/Services.h"

#include <algorithm>
#include <iostream>
#include <map>
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcovered-switch-default"
#endif
#include <nlohmann/json.hpp>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include <stdexcept>
#include <string>

using namespace esi;

void printInfo(std::ostream &os, AcceleratorConnection &acc, bool details);
void printHier(std::ostream &os, AcceleratorConnection &acc, bool details);
void printTelemetry(std::ostream &os, AcceleratorConnection &acc);
void printTelemetryJson(std::ostream &os, AcceleratorConnection &acc);

int main(int argc, const char *argv[]) {
  CliParser cli("esiquery");
  cli.description("Query an ESI system for information from the manifest.");

  CLI::App *versionSub =
      cli.add_subcommand("version", "Print ESI system version");
  bool infoDetails = false;
  CLI::App *infoSub =
      cli.add_subcommand("info", "Print ESI system information");
  infoSub->add_flag("--details", infoDetails,
                    "Print detailed information about the system");
  bool hierDetails = false;
  CLI::App *hierSub = cli.add_subcommand("hier", "Print ESI system hierarchy");
  hierSub->add_flag("--details", hierDetails,
                    "Print detailed information about the system");
  bool telemetryJson = false;
  CLI::App *telemetrySub =
      cli.add_subcommand("telemetry", "Print ESI system telemetry information");
  telemetrySub->add_flag("--json", telemetryJson,
                         "Dump telemetry information as JSON");

  if (int rc = cli.esiParse(argc, argv))
    return rc;
  if (!cli.get_help_ptr()->empty())
    return 0;

  Context &ctxt = cli.getContext();
  try {
    AcceleratorConnection *acc = cli.connect();
    const auto &info = *acc->getService<services::SysInfo>();

    if (*versionSub)
      std::cout << info.getEsiVersion() << std::endl;
    else if (*infoSub)
      printInfo(std::cout, *acc, infoDetails);
    else if (*hierSub)
      printHier(std::cout, *acc, hierDetails);
    else if (*telemetrySub) {
      if (telemetryJson)
        printTelemetryJson(std::cout, *acc);
      else
        printTelemetry(std::cout, *acc);
    }
    return 0;
  } catch (std::exception &e) {
    ctxt.getLogger().error("esiquery", e.what());
    return -1;
  }
}

void printInfo(std::ostream &os, AcceleratorConnection &acc, bool details) {
  std::string jsonManifest =
      acc.getService<services::SysInfo>()->getJsonManifest();
  Manifest m(acc.getCtxt(), jsonManifest);
  os << "API version: " << m.getApiVersion() << std::endl << std::endl;
  os << "********************************" << std::endl;
  os << "* Module information" << std::endl;
  os << "********************************" << std::endl;
  os << std::endl;
  for (ModuleInfo mod : m.getModuleInfos())
    os << "- " << mod;

  if (!details)
    return;

  os << std::endl;
  os << "********************************" << std::endl;
  os << "* Type table" << std::endl;
  os << "********************************" << std::endl;
  os << std::endl;
  size_t i = 0;
  for (const Type *t : m.getTypeTable())
    os << "  " << i++ << ": " << t->getID() << std::endl;
}

static bool showPort(const BundlePort &port, bool details) {
  return details ||
         (!port.getID().name.starts_with("__") && !port.getChannels().empty());
}

void printPort(std::ostream &os, const BundlePort &port, std::string indent,
               bool details) {
  if (!showPort(port, details))
    return;
  os << indent << "  " << port.getID() << ":";
  if (auto svcPort = dynamic_cast<const services::ServicePort *>(&port))
    if (auto svcPortStr = svcPort->toString(true)) {
      os << " " << *svcPortStr << std::endl;
      return;
    }
  os << std::endl;
  for (const auto &[name, chan] : port.getChannels())
    os << indent << "    " << name << ": " << chan.getType()->toString(true)
       << std::endl;
}

void printInstance(std::ostream &os, const HWModule *d, std::string indent,
                   bool details) {
  bool hasPorts =
      std::any_of(d->getPorts().begin(), d->getPorts().end(),
                  [&](const std::pair<const AppID, BundlePort &> port) {
                    return showPort(port.second, details);
                  });
  if (!details && !hasPorts && d->getChildren().empty())
    return;
  os << indent << "* Instance: ";
  if (auto inst = dynamic_cast<const Instance *>(d)) {
    os << inst->getID() << std::endl;
    if (inst->getInfo() && inst->getInfo()->name)
      os << indent << "* Module: " << *inst->getInfo()->name << std::endl;
  } else {
    os << "top" << std::endl;
  }

  os << indent << "* Ports:" << std::endl;
  for (const BundlePort &port : d->getPortsOrdered())
    printPort(os, port, indent + "  ", details);
  std::vector<const Instance *> children = d->getChildrenOrdered();
  if (!children.empty()) {
    os << indent << "* Children:" << std::endl;
    for (const Instance *child : d->getChildrenOrdered())
      printInstance(os, child, indent + "  ", details);
  }
  os << std::endl;
}

void printHier(std::ostream &os, AcceleratorConnection &acc, bool details) {
  Manifest manifest(acc.getCtxt(),
                    acc.getService<services::SysInfo>()->getJsonManifest());
  Accelerator *design = manifest.buildAccelerator(acc);
  os << "********************************" << std::endl;
  os << "* Design hierarchy" << std::endl;
  os << "********************************" << std::endl;
  os << std::endl;
  printInstance(os, design, /*indent=*/"", details);
}

// Recursively collect telemetry metrics into a hierarchical JSON structure.
static bool collectTelemetryJson(const HWModule &module, nlohmann::json &node) {
  bool hasData = false;

  for (const auto &portRef : module.getPortsOrdered()) {
    BundlePort &port = portRef.get();
    if (auto *metric =
            dynamic_cast<services::TelemetryService::Metric *>(&port)) {
      metric->connect();
      node[metric->getID().toString()] = metric->readInt();
      hasData = true;
    }
  }

  for (const Instance *child : module.getChildrenOrdered()) {
    nlohmann::json childNode = nlohmann::json::object();
    if (collectTelemetryJson(*child, childNode)) {
      node[child->getID().toString()] = childNode;
      hasData = true;
    }
  }

  return hasData;
}

void printTelemetryJson(std::ostream &os, AcceleratorConnection &acc) {
  Manifest manifest(acc.getCtxt(),
                    acc.getService<services::SysInfo>()->getJsonManifest());
  auto accel = manifest.buildAccelerator(acc);
  acc.getServiceThread()->addPoll(*accel);

  nlohmann::json root = nlohmann::json::object();
  if (!collectTelemetryJson(*accel, root))
    root = nlohmann::json{{"error", "No telemetry metrics found"}};

  os << root.dump(2) << std::endl;
}

void printTelemetry(std::ostream &os, AcceleratorConnection &acc) {
  Manifest manifest(acc.getCtxt(),
                    acc.getService<services::SysInfo>()->getJsonManifest());
  auto accel = manifest.buildAccelerator(acc);
  acc.getServiceThread()->addPoll(*accel);

  auto *telemetry = acc.getService<services::TelemetryService>();
  if (!telemetry) {
    os << "No telemetry service found" << std::endl;
    return;
  }
  os << "********************************" << std::endl;
  os << "* Telemetry" << std::endl;
  os << "********************************" << std::endl;
  os << std::endl;

  const std::map<AppIDPath, services::TelemetryService::Metric *>
      &telemetryPorts = telemetry->getTelemetryPorts();
  for (const auto &[id, port] : telemetryPorts) {
    port->connect();
    os << id << ": ";
    os.flush();
    uint64_t value = port->readInt();
    os << value << std::endl;
  }
}
