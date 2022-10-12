//===- PrettyPrinterBuilder.h - Pretty printing builder -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper classes for using PrettyPrinter.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_PRETTYPRINTERBUILDER_H
#define CIRCT_SUPPORT_PRETTYPRINTERBUILDER_H

#include "circt/Support/PrettyPrinter.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"

namespace circt {
namespace pretty {

//===----------------------------------------------------------------------===//
// Convenience builders.
//===----------------------------------------------------------------------===//

class PPBuilder : protected PrettyPrinter::Listener {
  PrettyPrinter pp;

public:
  PPBuilder(llvm::raw_ostream &os, uint32_t margin) : pp(os, margin, this){};

  /// Add new token.
  template <typename T, typename... Args>
  typename std::enable_if_t<std::is_base_of_v<Token, T>> add(Args &&...args) {
    pp.add(T(std::forward<Args>(args)...));
  }
  void addToken(Token &t) { pp.add(t); }

  /// End of a stream.
  void eof() { pp.eof(); }

  /// Add a literal (with external storage).
  void literal(StringRef str) { add<StringToken>(str); }

  /// Add a non-breaking space.
  void nbsp() { literal(" "); }

  /// Add a newline (break too wide to fit, always breaks).
  void newline() { add<BreakToken>(PrettyPrinter::kInfinity); }

  /// End a group.
  void end() { add<EndToken>(); }

  /// Add breakable spaces.
  void spaces(uint32_t n) { add<BreakToken>(n); }

  /// Add a breakable space.
  void space() { spaces(1); }

  /// Add a break that is zero-wide if not broken.
  void zerobreak() { add<BreakToken>(0); }

  /// Start a consistent group with specified offset.
  void cbox(int32_t offset = 0) { add<BeginToken>(offset, Breaks::Consistent); }

  /// Start an inconsistent group with specified offset.
  void ibox(int32_t offset = 0) {
    add<BeginToken>(offset, Breaks::Inconsistent);
  }

  /// Open a cbox that closes when returned object goes out of scope.
  [[nodiscard]] auto scopedCBox(int32_t offset = 0) {
    cbox(offset);
    return llvm::make_scope_exit([&]() { end(); });
  }

  /// Open an ibox that closes when returned object goes out of scope.
  [[nodiscard]] auto scopedIBox(int32_t offset = 0) {
    ibox(offset);
    return llvm::make_scope_exit([&]() { end(); });
  }
};

/// Variant that saves strings that are live in the pretty-printer.
/// Once they're no longer referenced, memory is reset.
/// Allows differentiating between strings to save and external strings.
class PPBuilderStringSaver : public PPBuilder {
  llvm::BumpPtrAllocator alloc;
  llvm::StringSaver strings;

public:
  PPBuilderStringSaver(llvm::raw_ostream &os, uint32_t margin)
      : PPBuilder(os, margin), strings(alloc){};

  /// Add string, save in storage.
  void savedWord(StringRef str) { add<StringToken>(strings.save(str)); }

protected:
  /// PrettyPrinter::Listener::clear -- indicates no external refs.
  void clear() override;
};

//===----------------------------------------------------------------------===//
// Streaming support.
//===----------------------------------------------------------------------===//

/// Send one of these to PPStream to add the corresponding token.
/// See PPBuilder for details of each.
enum class PP {
  space,
  nbsp,
  newline,
  ibox0,
  ibox2,
  cbox0,
  cbox2,
  end,
  zerobreak,
  eof
};

/// String wrapper to indicate string has external storage.
struct PPExtString {
  StringRef str;
  explicit PPExtString(StringRef str) : str(str) {}
};

/// String wrapper to indicate string needs to be saved.
struct PPSaveString {
  StringRef str;
  explicit PPSaveString(StringRef str) : str(str) {}
};

class PPStream : public PPBuilderStringSaver {
public:
  PPStream(llvm::raw_ostream &os, uint32_t margin)
      : PPBuilderStringSaver(os, margin) {}

  /// Add a string literal (external storage).
  PPStream &operator<<(const char *s) {
    literal(s);
    return *this;
  }
  /// Add a string token (saved to storage).
  PPStream &operator<<(StringRef s) {
    savedWord(s);
    return *this;
  }

  /// String has external storage.
  PPStream &operator<<(PPExtString &&str) {
    literal(str.str);
    return *this;
  }

  /// String must be saved.
  PPStream &operator<<(PPSaveString &&str) {
    savedWord(str.str);
    return *this;
  }

  /// Convenience for inline streaming of builder methods.
  PPStream &operator<<(PP s) {
    switch (s) {
    case PP::space:
      space();
      break;
    case PP::nbsp:
      nbsp();
      break;
    case PP::newline:
      newline();
      break;
    case PP::ibox0:
      ibox();
      break;
    case PP::ibox2:
      ibox(2);
      break;
    case PP::cbox0:
      cbox();
      break;
    case PP::cbox2:
      cbox(2);
      break;
    case PP::end:
      end();
      break;
    case PP::zerobreak:
      zerobreak();
      break;
    case PP::eof:
      eof();
      break;
    }
    return *this;
  }

  /// Stream support for user-created Token's.
  PPStream &operator<<(Token &t) {
    addToken(t);
    return *this;
  }

  /// Write escaped versions of the string, saved in storage.
  PPStream &writeEscaped(StringRef str, bool useHexEscapes = false) {
    return writeQuotedEscaped(str, useHexEscapes, "", "");
  }
  PPStream &writeQuotedEscaped(StringRef str, bool useHexEscapes = false,
                               StringRef left = "\"", StringRef right = "\"");
};

} // end namespace pretty
} // end namespace circt

#endif // CIRCT_SUPPORT_PRETTYPRINTERBUILDER_H
