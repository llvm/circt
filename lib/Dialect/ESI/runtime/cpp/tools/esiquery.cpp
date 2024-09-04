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
#include "esi/Manifest.h"
#include "esi/Services.h"

#include <iostream>
#include <map>
#include <stdexcept>

using namespace esi;

void printInfo(std::ostream &os, AcceleratorConnection &acc);
void printHier(std::ostream &os, AcceleratorConnection &acc);

int main(int argc, const char *argv[]) {
  // TODO: find a command line parser library rather than doing this by hand.
  if (argc < 3) {
    std::cerr << "Expected usage: " << argv[0]
              << " <backend> <connection specifier> [command]" << std::endl;
    return -1;
  }

  Context ctxt;

  const char *backend = argv[1];
  const char *conn = argv[2];
  std::string cmd;
  if (argc > 3)
    cmd = argv[3];

  try {
    std::unique_ptr<AcceleratorConnection> acc = ctxt.connect(backend, conn);
    const auto &info = *acc->getService<services::SysInfo>();

    if (cmd == "version")
      std::cout << "ESI system version: " << info.getEsiVersion() << std::endl;
    else if (cmd == "json_manifest")
      std::cout << info.getJsonManifest() << std::endl;
    else if (cmd == "info")
      printInfo(std::cout, *acc);
    else if (cmd == "hier")
      printHier(std::cout, *acc);
    else {
      std::cout << "Connection successful." << std::endl;
      if (!cmd.empty()) {
        std::cerr << "Unknown command: " << cmd << std::endl;
        return 1;
      }
    }
    return 0;
  } catch (std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
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
  printInstance(os, design);
}
