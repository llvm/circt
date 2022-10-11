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
//   to accomodate longer widths when needed.
//   Since scanStack references buffered tokens by index, we track an offset
//   that we increase when dropping off the front.
//   When the scan stack is cleared the bufffer is reset, including this offset.
// * Optionally, minimum amount of space is granted regardless of indentation.
//   To avoid forcing expressions against the line limit, never try to print
//   an expression in, say, 2 columns, as this is unlikely to produce good
//   output.
//   (TODO)
// * Indentation tracked from left not relative to margin (linewidth).
//   (TODO)
// * Indentation emitted lazily, avoid trailing whitespace.
//   (TODO)
//
// There are many pretty-printing implementations based on this paper,
// and research literature is rich with functional formulations based originally
// on this algorithm.
//
// Implementations of note that have interesting modifications for their
// languages and modernization of the paper's algorithm.
// * prettyplease / rustc_ast_pretty
//   Pretty-printers for rust, the first being useful for rustfmt-like output.
//   These have largely the same code and were based on one another.
//     https://github.com/dtolnay/prettyplease
//     https://github.com/rust-lang/rust/tree/master/compiler/rustc_ast_pretty
//   This is closest to the paper's algorithm with modernizations,
//   and most of the initial tweaks also implemented here (thanks!).
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

// TODO: clear token storage when possible (left=right, maybe scanStack empty).
//       also reset offsets/cursors.

namespace circt {
namespace pretty {

// TODO: parameter or not at all
// auto constexpr MIN_SPACE = 20;

auto constexpr debug = false;

/// Add token for printing.  In Oppen, this is "scan".
void PrettyPrinter::add(Token &t) {
  llvm::TypeSwitch<Token *, void>(&t)
      .Case([&](StringToken *s) {
        // If nothing on stack, directly print
        FormattedToken f{t, (sint)s->text.size()};
        if (scanStack.empty())
          return print(f);
        tokens.push_back(f);
        rightTotal += f.size;
        checkStream();
      })
      .Case([&](BreakToken *b) {
        if (scanStack.empty()) {
          leftTotal = rightTotal = 1;
          tokens.clear();
          tokenOffset = 0;
        } else {
          // Update sizes of prev begin/break/end
          checkStack(0);
        }
        tokens.push_back({t, -rightTotal});
        scanStack.push_back(tokenOffset + tokens.size() - 1);
        rightTotal += b->spaces;
      })
      .Case([&](BeginToken *b) {
        if (scanStack.empty()) {
          leftTotal = rightTotal = 1;
          tokens.clear();
          tokenOffset = 0;
        }
        tokens.push_back({t, -rightTotal});
        scanStack.push_back(tokenOffset + tokens.size() - 1);
      })
      .Case([&](EndToken *end) {
        if (scanStack.empty())
          return print({t, 0});
        tokens.push_back({t, -1});
        scanStack.push_back(tokenOffset + tokens.size() - 1);
      });
}

/// Break encountered, set sizes of begin/breaks in scanStack that we now know.
void PrettyPrinter::checkStack(uint depth) {
  if (scanStack.empty())
    return;
  auto x = scanStack.back();
  auto &t = tokens[x - tokenOffset];
  if (auto *b = llvm::dyn_cast<BeginToken>(&t.token)) {
    if (depth > 0) {
      scanStack.pop_back();
      t.size += rightTotal;
      checkStack(depth - 1);
    }
  } else if (auto *e = llvm::dyn_cast<EndToken>(&t.token)) {
    scanStack.pop_back();
    t.size = 1;
    checkStack(depth + 1);
  } else {
    // break, not string (?)
    scanStack.pop_back();
    t.size += rightTotal;
    if (depth > 0)
      checkStack(depth);
  }
}

/// Check if there's enough tokens to hit width, if so print.
/// If scan size is wider than line, it's infinity.
void PrettyPrinter::checkStream() {
  // If there's more space, do nothing.
  if (rightTotal - leftTotal <= space || scanStack.empty())
    return;

  // While buffer needs more than 1 line to print, print and consume.

  // Ran out of space, set size to infinity and take off scan stack.
  // No need to keep track as we know enough to know this won't fit.
  if (tokenOffset == scanStack.front()) {
    tokens.front().size = 0xFFFFF; // ~sint{0}; // INFINITY
    scanStack.pop_front();
  }
  advanceLeft();
  // TODO: don't recurse
  if (!tokens.empty())
    checkStream();
}

/// Print out tokens we know sizes for, and drop from token buffer.
void PrettyPrinter::advanceLeft() {
  assert(!tokens.empty());

  while (!tokens.empty() && tokens.front().size >= 0) {
    auto t = tokens.front();
    tokens.pop_front();
    ++tokenOffset;

    assert(&t.token);
    print(t);
    leftTotal += llvm::TypeSwitch<Token *, sint>(&t.token)
                     .Case([&](BreakToken *b) { return b->spaces; })
                     .Case([&](StringToken *s) { return s->text.size(); })
                     .Default([](auto *) { return 0; });
  }
}

/// Print a token, maintaining printStack for context.
void PrettyPrinter::print(FormattedToken f) {
  llvm::TypeSwitch<Token *, void>(&f.token)
      .Case([&](StringToken *s) {
        space -= f.size;
        os << s->text;
      })
      .Case([&](BreakToken *b) {
        // If nothing on print stack (no begin context),
        // wrap w/no offset and emit greedily.
        PrintEntry outer{margin, PrintBreaks::Inconsistent};
        auto &frame = printStack.empty() ? outer : printStack.back();
        bool fits =
            frame.breaks == PrintBreaks::Fits ||
            (frame.breaks == PrintBreaks::Inconsistent && f.size <= space);
        if (fits) {
          space -= b->spaces;
          os.indent(b->spaces);
        } else {
          if (debug) {
            if (space)
              os << "┆";
            if (space > 2)
              os.indent(space - 2);
            if (space > 1)
              os << "┇";
          }
          os << "\n";
          space = frame.offset - b->offset;
          os.indent(std::max<ssize_t>(ssize_t(margin) - space, 0));
        }
      })
      .Case([&](BeginToken *b) {
        if (f.size > space) {
          auto breaks = b->breaks == Breaks::Consistent
                            ? PrintBreaks::Consistent
                            : PrintBreaks::Inconsistent;
          printStack.push_back({uint(space - b->offset), breaks});
        } else {
          printStack.push_back({0, PrintBreaks::Fits});
        }
      })
      .Case([&](EndToken *) {
        printStack.pop_back(); // breaks
      });
}
} // end namespace pretty
} // end namespace circt
