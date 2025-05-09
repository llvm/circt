//===- CLI.h - ESI runtime tool CLI parser common ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the common CLI parser code for ESI runtime tools. Exposed
// publicly so that out-of-tree tools can use it. This is a header-only library
// to make compilation easier for out-of-tree tools.
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT (lib/dialect/ESI/runtime/cpp).
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_CLI_H
#define ESI_CLI_H

#include "CLI/CLI.hpp"
#include "esi/Context.h"

namespace esi {

/// Common options and code for ESI runtime tools.
class CliParser : public CLI::App {
public:
  CliParser(const std::string &toolName)
      : CLI::App(toolName), debug(false), verbose(false) {
    add_option("backend", backend, "Backend to use for connection")->required();
    add_option("connection", connStr,
               "Connection string to use for accelerator communication")
        ->required();
    add_flag("--debug", debug, "Enable debug logging");
#ifdef ESI_RUNTIME_TRACE
    add_flag("--trace", trace, "Enable trace logging");
#endif
    add_flag("-v,--verbose", verbose, "Enable verbose (info) logging");
    require_subcommand(0, 1);
  }

  /// Run the parser.
  int esiParse(int argc, const char **argv) {
    CLI11_PARSE(*this, argc, argv);
    if (trace)
      ctxt = Context::withLogger<ConsoleLogger>(Logger::Level::Trace);
    else if (debug)
      ctxt = Context::withLogger<ConsoleLogger>(Logger::Level::Debug);
    else if (verbose)
      ctxt = Context::withLogger<ConsoleLogger>(Logger::Level::Info);
    return 0;
  }

  /// Connect to the accelerator using the specified backend and connection.
  std::unique_ptr<AcceleratorConnection> connect() {
    return ctxt.connect(backend, connStr);
  }

  /// Get the context.
  Context &getContext() { return ctxt; }

protected:
  Context ctxt;

  std::string backend;
  std::string connStr;
  bool trace = false;
  bool debug = false;
  bool verbose = false;
};

} // namespace esi

#endif // ESI_CLI_H
