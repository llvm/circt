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

#include <iostream>
#include <map>
#include <stdexcept>

using namespace esi;

void printInfo(std::ostream &os, AcceleratorConnection &acc);
void printHier(std::ostream &os, AcceleratorConnection &acc);
void printTelemetry(std::ostream &os, AcceleratorConnection &acc);

int main(int argc, const char *argv[]) {
  CliParser cli("esiquery");
  cli.description("Query an ESI system for information from the manifest.");

  CLI::App *versionSub =
      cli.add_subcommand("version", "Print ESI system version");
  CLI::App *infoSub =
      cli.add_subcommand("info", "Print ESI system information");
  CLI::App *hierSub = cli.add_subcommand("hier", "Print ESI system hierarchy");
  CLI::App *telemetrySub =
      cli.add_subcommand("telemetry", "Print ESI system telemetry information");

  if (int rc = cli.esiParse(argc, argv))
    return rc;
  if (!cli.get_help_ptr()->empty())
    return 0;

  Context &ctxt = cli.getContext();
  try {
    std::unique_ptr<AcceleratorConnection> acc = cli.connect();
    const auto &info = *acc->getService<services::SysInfo>();

    if (*versionSub)
      std::cout << info.getEsiVersion() << std::endl;
    else if (*infoSub)
      printInfo(std::cout, *acc);
    else if (*hierSub)
      printHier(std::cout, *acc);
    else if (*telemetrySub)
      printTelemetry(std::cout, *acc);
    return 0;
  } catch (std::exception &e) {
    ctxt.getLogger().error("esiquery", e.what());
    return -1;
  }
}

void printInfo(std::ostream &os, AcceleratorConnection &acc) {
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

  os << std::endl;
  os << "********************************" << std::endl;
  os << "* Type table" << std::endl;
  os << "********************************" << std::endl;
  os << std::endl;
  size_t i = 0;
  for (const Type *t : m.getTypeTable())
    os << "  " << i++ << ": " << t->getID() << std::endl;
}

void printPort(std::ostream &os, const BundlePort &port,
               std::string indent = "") {
  os << indent << "  " << port.getID() << ":";
  if (auto svcPort = dynamic_cast<const services::ServicePort *>(&port))
    if (auto svcPortStr = svcPort->toString()) {
      os << " " << *svcPortStr << std::endl;
      return;
    }
  os << std::endl;
  for (const auto &[name, chan] : port.getChannels())
    os << indent << "    " << name << ": " << chan.getType()->getID()
       << std::endl;
}

void printInstance(std::ostream &os, const HWModule *d,
                   std::string indent = "") {
  os << indent << "* Instance:";
  if (auto inst = dynamic_cast<const Instance *>(d))
    os << inst->getID() << std::endl;
  else
    os << "top" << std::endl;
  os << indent << "* Ports:" << std::endl;
  for (const BundlePort &port : d->getPortsOrdered())
    printPort(os, port, indent + "  ");
  std::vector<const Instance *> children = d->getChildrenOrdered();
  if (!children.empty()) {
    os << indent << "* Children:" << std::endl;
    for (const Instance *child : d->getChildrenOrdered())
      printInstance(os, child, indent + "  ");
  }
  os << std::endl;
}

void printHier(std::ostream &os, AcceleratorConnection &acc) {
  Manifest manifest(acc.getCtxt(),
                    acc.getService<services::SysInfo>()->getJsonManifest());
  Accelerator *design = manifest.buildAccelerator(acc);
  os << "********************************" << std::endl;
  os << "* Design hierarchy" << std::endl;
  os << "********************************" << std::endl;
  os << std::endl;
  printInstance(os, design, /*indent=*/"");
}

void printTelemetry(std::ostream &os, AcceleratorConnection &acc) {
  Manifest manifest(acc.getCtxt(),
                    acc.getService<services::SysInfo>()->getJsonManifest());
  auto accel = manifest.buildAccelerator(acc);
  acc.getServiceThread()->addPoll(*accel);

  auto telemetry = acc.getService<services::TelemetryService>();
  if (!telemetry) {
    os << "No telemetry service found" << std::endl;
    return;
  }
  os << "********************************" << std::endl;
  os << "* Telemetry" << std::endl;
  os << "********************************" << std::endl;
  os << std::endl;

  const std::map<AppIDPath, services::TelemetryService::Telemetry *>
      &telemetryPorts = telemetry->getTelemetryPorts();
  for (const auto &[id, port] : telemetryPorts) {
    port->connect();
    os << id << ": ";
    os.flush();
    uint64_t value = *port->read().get().as<uint64_t>();
    os << value << std::endl;
  }
}
