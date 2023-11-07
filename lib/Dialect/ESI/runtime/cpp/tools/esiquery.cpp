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
#include "esi/StdServices.h"

#include <iostream>
#include <map>
#include <stdexcept>

using namespace esi;

void printInfo(std::ostream &os, Accelerator &acc);

int main(int argc, const char *argv[]) {
  // TODO: find a command line parser library rather than doing this by hand.
  if (argc < 3) {
    std::cerr << "Expected usage: " << argv[0]
              << " <backend> <connection specifier> [command]" << std::endl;
    return -1;
  }

  const char *backend = argv[1];
  const char *conn = argv[2];
  std::string cmd;
  if (argc > 3)
    cmd = argv[3];

  try {
    std::unique_ptr<Accelerator> acc = registry::connect(backend, conn);
    const auto &info = *acc->getService<services::SysInfo>();

    if (cmd == "version")
      std::cout << "ESI system version: " << info.esiVersion() << std::endl;
    else if (cmd == "json_manifest")
      std::cout << info.jsonManifest() << std::endl;
    else if (cmd == "info")
      printInfo(std::cout, *acc);
    // TODO: add a command to print out the instance hierarchy.
    else
      std::cout << "Connection successful." << std::endl;

    if (cmd.empty())
      return 0;
    std::cerr << "Unknown command: " << cmd << std::endl;
    return 1;

  } catch (std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }
}

void printInfo(std::ostream &os, Accelerator &acc) {
  std::string jsonManifest =
      acc.getService<services::SysInfo>()->jsonManifest();
  Manifest m(jsonManifest);
  os << "API version: " << m.apiVersion() << std::endl << std::endl;
  os << "********************************" << std::endl;
  os << "* Design information" << std::endl;
  os << "********************************" << std::endl;
  os << std::endl;
  for (ModuleInfo mod : m.moduleInfos())
    os << mod << std::endl;
}
