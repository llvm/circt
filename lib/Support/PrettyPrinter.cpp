//===- PrettyPrinter.h - Pretty printing ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements a pretty-printer.
// "PrettyPrinting", Derek C. Oppen, 1980.
// https://dx.doi.org/10.1145/357114.357115
//
// This was selected as it is linear in number of tokens O(n) and requires
// memory O(linewidth).
//
// This has been adjusted from the paper:
// * Deque for tokens instead of ringbuffer + left/right cursors.
//   This is simpler to reason about and allows us to easily grow the buffer
//   to accommodate longer widths when needed (and not reserve 3*linewidth).
//   Since scanStack references buffered tokens by index, we track an offset
//   that we increase when dropping off the front.
//   When the scan stack is cleared the buffer is reset, including this offset.
// * Indentation tracked from left not relative to margin (linewidth).
// * Indentation emitted lazily, avoid trailing whitespace.
// * Group indentation styles: Visual and Block, set on 'begin' tokens.
//   "Visual" is the style in the paper, offset relative to current column.
//   "Block" is relative to current base indentation.
// * Optionally, minimum amount of space is granted regardless of indentation.
//   To avoid forcing expressions against the line limit, never try to print
//   an expression in, say, 2 columns, as this is unlikely to produce good
//   output.
//   (TODO)
// * Neverbreak. (TODO: explain/motivate).
//
//
// There are many pretty-printing implementations based on this paper,
// and research literature is rich with functional formulations based originally
// on this algorithm.
//
// Implementations of note that have interesting modifications for their
// languages and modernization of the paper's algorithm:
// * prettyplease / rustc_ast_pretty
//   Pretty-printers for rust, the first being useful for rustfmt-like output.
//   These have largely the same code and were based on one another.
//     https://github.com/dtolnay/prettyplease
//     https://github.com/rust-lang/rust/tree/master/compiler/rustc_ast_pretty
//   This is closest to the paper's algorithm with modernizations,
//   and most of the initial tweaks have also been implemented here (thanks!).
// * swift-format: https://github.com/apple/swift-format/
//
// If we want fancier output or need to handle more complicated constructs,
// both are good references for lessons and ideas.
//
// FWIW, at time of writing these have compatible licensing (Apache 2.0).
//
//===----------------------------------------------------------------------===//

#include "circt/Support/PrettyPrinter.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace pretty {

/// Destructor, anchor.
PrettyPrinter::Listener::~Listener() = default;

/// Add token for printing.  In Oppen, this is "scan".
void PrettyPrinter::add(Token t) {
  // Add token to tokens, and add its index to scanStack.
  auto addScanToken = [&](auto offset) {
    auto right = tokenOffset + tokens.size();
    scanStack.push_back(right);
    tokens.push_back({t, offset});
  };
  llvm::TypeSwitch<Token *, void>(&t)
      .Case([&](StringToken *s) {
        // If nothing on stack, directly print
        FormattedToken f{t, (int32_t)s->text().size()};
        // Empty string token isn't /wrong/ but can have unintended effect.
        assert(!s->text().empty() && "empty string token");
        if (scanStack.empty())
          return print(f);
        tokens.push_back(f);
        rightTotal += f.size;
        checkStream();
      })
      .Case([&](BreakToken *b) {
        if (scanStack.empty())
          clear();
        else
          checkStack();
        addScanToken(-rightTotal);
        rightTotal += b->spaces();
      })
      .Case([&](BeginToken *b) {
        if (scanStack.empty())
          clear();
        addScanToken(-rightTotal);
      })
      .Case([&](EndToken *end) {
        if (scanStack.empty())
          return print({t, 0});
        addScanToken(-1);
      });
}

void PrettyPrinter::eof() {
  if (!scanStack.empty()) {
    checkStack();
    advanceLeft();
  }
  assert(scanStack.empty() && "unclosed groups at EOF");
  if (scanStack.empty())
    clear();
}

void PrettyPrinter::clear() {
  assert(scanStack.empty() && "clearing tokens while still on scan stack");
  leftTotal = rightTotal = 1;
  assert(tokens.empty());
  tokens.clear();
  tokenOffset = 0;
  if (listener)
    listener->clear();
}

/// Break encountered, set sizes of begin/breaks in scanStack that we now know.
void PrettyPrinter::checkStack() {
  unsigned depth = 0;
  while (!scanStack.empty()) {
    auto x = scanStack.back();
    assert(x >= tokenOffset && tokens.size() + tokenOffset > x);
    auto &t = tokens[x - tokenOffset];
    if (auto *b = llvm::dyn_cast<BeginToken>(&t.token)) {
      if (depth == 0)
        break;
      scanStack.pop_back();
      t.size += rightTotal;
      --depth;
    } else if (auto *e = llvm::dyn_cast<EndToken>(&t.token)) {
      scanStack.pop_back();
      t.size = 1;
      ++depth;
    } else {
      scanStack.pop_back();
      t.size += rightTotal;
      if (depth == 0)
        break;
    }
  }
}

/// Check if there's enough tokens to hit width, if so print.
/// If scan size is wider than line, it's infinity.
void PrettyPrinter::checkStream() {
  // While buffer needs more than 1 line to print, print and consume.
  assert(!tokens.empty());
  assert(leftTotal >= 0);
  assert(rightTotal >= 0);
  while (rightTotal - leftTotal > space && !tokens.empty()) {

    // Ran out of space, set size to infinity and take off scan stack.
    // No need to keep track as we know enough to know this won't fit.
    if (!scanStack.empty() && tokenOffset == scanStack.front()) {
      tokens.front().size = kInfinity;
      scanStack.pop_front();
    }
    advanceLeft();
  }
}

/// Print out tokens we know sizes for, and drop from token buffer.
void PrettyPrinter::advanceLeft() {
  assert(!tokens.empty());

  while (!tokens.empty() && tokens.front().size >= 0) {
    auto f = tokens.front();
    tokens.pop_front();
    ++tokenOffset;

    print(f);
    leftTotal += llvm::TypeSwitch<Token *, int32_t>(&f.token)
                     .Case([&](BreakToken *b) { return b->spaces(); })
                     .Case([&](StringToken *s) { return s->text().size(); })
                     .Default([](auto *) { return 0; });
  }
}

/// Print a token, maintaining printStack for context.
void PrettyPrinter::print(FormattedToken f) {
  llvm::TypeSwitch<Token *, void>(&f.token)
      .Case([&](StringToken *s) {
        space -= f.size;
        os.indent(pendingIndentation);
        pendingIndentation = 0;
        os << s->text();
      })
      .Case([&](BreakToken *b) {
        auto &frame = getPrintFrame();
        bool fits =
            b->neverbreak() || frame.breaks == PrintBreaks::Fits ||
            (frame.breaks == PrintBreaks::Inconsistent && f.size <= space);
        if (fits) {
          space -= b->spaces();
          pendingIndentation += b->spaces();
        } else {
          os << "\n";
          pendingIndentation =
              std::max<ssize_t>(ssize_t{indent} + b->offset(), 0);
          space = std::max<ssize_t>(ssize_t{margin} - pendingIndentation, 0);
        }
      })
      .Case([&](BeginToken *b) {
        if (f.size > space) {
          auto breaks = b->breaks() == Breaks::Consistent
                            ? PrintBreaks::Consistent
                            : PrintBreaks::Inconsistent;
          if (b->style() == IndentStyle::Visual)
            indent = std::max<ssize_t>(ssize_t{margin} - space, 0);
          indent += b->offset();
          printStack.push_back({indent, breaks});
        } else {
          printStack.push_back({0, PrintBreaks::Fits});
        }
      })
      .Case([&](EndToken *) {
        assert(!printStack.empty() && "more ends than begins?");
        // Try to tolerate this when assertions are disabled.
        if (printStack.empty())
          return;
        printStack.pop_back();
        auto &frame = getPrintFrame();
        if (frame.breaks != PrintBreaks::Fits)
          indent = frame.offset;
      });
}
} // end namespace pretty
} // end namespace circt
