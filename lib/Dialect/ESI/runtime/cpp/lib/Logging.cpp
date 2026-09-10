//===- Logging.cpp - ESI logging system API implementation ----------------===//
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

#include "esi/Logging.h"
#include "esi/Common.h"

#include "fmt/color.h"
#include "fmt/core.h"

#include <iostream>
#include <mutex>

using namespace esi;

void TSLogger::log(Level level, const std::string &subsystem,
                   const std::string &msg,
                   const std::map<std::string, std::any> *details) {
  std::scoped_lock<std::mutex> lock(mutex);
  logImpl(level, subsystem, msg, details);
}

StreamLogger::StreamLogger(Level minLevel)
    : TSLogger(minLevel <= Level::Debug, minLevel <= Level::Trace),
      minLevel(minLevel), outStream(std::cout), errorStream(std::cerr) {}

void StreamLogger::logImpl(Level level, const std::string &subsystem,
                           const std::string &msg,
                           const std::map<std::string, std::any> *details) {
  if (level < minLevel)
    return;
  std::ostream &os = level == Level::Error ? errorStream : outStream;
  unsigned indentSpaces = 0;

  switch (level) {
  case Level::Error:
    os << "[  ERROR] ";
    indentSpaces = 8;
    break;
  case Level::Warning:
    os << "[WARNING] ";
    indentSpaces = 10;
    break;
  case Level::Info:
    os << "[   INFO] ";
    indentSpaces = 7;
    break;
  case Level::Debug:
    os << "[  DEBUG] ";
    indentSpaces = 8;
    break;
  case Level::Trace:
    os << "[  TRACE] ";
    indentSpaces = 8;
    break;
  }

  if (!subsystem.empty()) {
    os << "[" << subsystem << "] ";
    indentSpaces += subsystem.size() + 3;
  }
  os << msg << std::endl;

  if (details) {
    std::string indent(indentSpaces, ' ');
    for (const auto &detail : *details)
      os << indent << detail.first << ": " << toString(detail.second) << "\n";
  }
  os.flush();
}

ConsoleLogger::ConsoleLogger(Level minLevel)
    : TSLogger(minLevel <= Level::Debug, minLevel <= Level::Trace),
      minLevel(minLevel) {}

void ConsoleLogger::logImpl(Level level, const std::string &subsystem,
                            const std::string &msg,
                            const std::map<std::string, std::any> *details) {
  if (level < minLevel)
    return;
  FILE *os = level == Level::Error ? stderr : stdout;
  unsigned indentSpaces = 0;

  switch (level) {
  case Level::Error:
    fmt::print(os, fmt::fg(fmt::color::red), "[  ERROR] ");
    indentSpaces = 8;
    break;
  case Level::Warning:
    fmt::print(os, fmt::fg(fmt::color::yellow), "[WARNING] ");
    indentSpaces = 10;
    break;
  case Level::Info:
    fmt::print(os, fmt::fg(fmt::color::dark_green), "[   INFO] ");
    indentSpaces = 7;
    break;
  case Level::Debug:
    fmt::print(os, fmt::fg(fmt::color::beige), "[  DEBUG] ");
    indentSpaces = 8;
    break;
  case Level::Trace:
    fmt::print(os, fmt::fg(fmt::color::burly_wood), "[  TRACE] ");
    indentSpaces = 8;
    break;
  }

  if (!subsystem.empty()) {
    fmt::print(os, "[{}]", subsystem);
    indentSpaces += subsystem.size() + 3;
  }
  fmt::print(os, " {}", msg);
  fmt::print(os, "\n");

  if (details) {
    std::string indent(indentSpaces, ' ');
    for (const auto &detail : *details)
      fmt::print(os, "{} {}: {}\n", indent, detail.first,
                 toString(detail.second));
  }
}

std::string esi::toString(const std::any &value) {
  if (value.type() == typeid(std::string))
    return std::any_cast<std::string>(value);
  if (value.type() == typeid(int))
    return std::to_string(std::any_cast<int>(value));
  if (value.type() == typeid(long))
    return std::to_string(std::any_cast<long>(value));
  if (value.type() == typeid(unsigned))
    return std::to_string(std::any_cast<unsigned>(value));
  if (value.type() == typeid(unsigned long))
    return std::to_string(std::any_cast<unsigned long>(value));
  if (value.type() == typeid(bool))
    return std::any_cast<bool>(value) ? "true" : "false";
  if (value.type() == typeid(double))
    return std::to_string(std::any_cast<double>(value));
  if (value.type() == typeid(float))
    return std::to_string(std::any_cast<float>(value));
  if (value.type() == typeid(const char *))
    return std::string(std::any_cast<const char *>(value));
  if (value.type() == typeid(char))
    return std::string(1, std::any_cast<char>(value));
  if (value.type() == typeid(MessageData))
    return std::any_cast<MessageData>(value).toHex();
  return "<unknown>";
}
