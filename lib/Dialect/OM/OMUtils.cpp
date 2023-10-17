//===- OMUtils.cpp - OM Utility Functions ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/OMUtils.h"
#include "circt/Dialect/OM/OMAttributes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace circt;
using namespace om;

namespace {
struct PathParser {
  enum class TokenKind {
    Space,
    Tilde,
    Bar,
    Colon,
    Greater,
    Slash,
    LBrace,
    RBrace,
    Period,
    Id,
    Eof
  };

  struct Token {
    TokenKind kind;
    StringRef spelling;
  };

  Token formToken(TokenKind kind, size_t n) {
    auto s = spelling.take_front(n);
    spelling = spelling.drop_front(n);
    return {kind, s};
  }

  bool isIDChar(char c) {
    return c != ' ' && c != '~' && c != '|' && c != ':' && c != '>' &&
           c != '/' && c != '[' && c != ']' && c != '.';
  }

  Token parseToken() {
    size_t pos = 0;
    auto size = spelling.size();
    if (0 == size)
      return formToken(TokenKind::Eof, pos);
    auto current = spelling[pos];
    switch (current) {
    case '~':
      return formToken(TokenKind::Tilde, pos + 1);
    case '|':
      return formToken(TokenKind::Bar, pos + 1);
    case ':':
      return formToken(TokenKind::Colon, pos + 1);
    case '>':
      return formToken(TokenKind::Greater, pos + 1);
    case '/':
      return formToken(TokenKind::Slash, pos + 1);
    case '[':
      return formToken(TokenKind::LBrace, pos + 1);
    case ']':
      return formToken(TokenKind::RBrace, pos + 1);
    case '.':
      return formToken(TokenKind::Period, pos + 1);
    default:
      break;
    }

    // Parsing a token.
    while (pos != size && isIDChar(spelling[pos]))
      ++pos;
    return formToken(TokenKind::Id, pos);
  }

  ParseResult parseToken(TokenKind kind, StringRef &result) {
    auto save = spelling;
    auto token = parseToken();
    if (token.kind != kind)
      return spelling = save, failure();
    result = token.spelling;
    return success();
  }

  ParseResult parseToken(TokenKind kind) {
    StringRef ignore;
    return parseToken(kind, ignore);
  }

  /// basepath ::= eof | id '/' id (':' id '/' id)* eof
  ParseResult parseBasePath(PathAttr &pathAttr) {
    if (succeeded(parseToken(TokenKind::Eof)))
      return success();
    SmallVector<PathElement> path;
    while (true) {
      StringRef module;
      StringRef instance;
      if (parseToken(TokenKind::Id, module) || parseToken(TokenKind::Slash) ||
          parseToken(TokenKind::Id, instance))
        return failure();
      path.emplace_back(StringAttr::get(context, module),
                        StringAttr::get(context, instance));
      if (parseToken(TokenKind::Colon))
        break;
    }
    pathAttr = PathAttr::get(context, path);
    if (parseToken(TokenKind::Eof))
      return failure();
    return success();
  }

  /// path ::= id ('/' id ':' path | ('>' id ('[' id ']' | '.' id)* )?) eof
  ParseResult parsePath(PathAttr &pathAttr, StringAttr &moduleAttr,
                        StringAttr &refAttr, StringAttr &fieldAttr) {
    SmallVector<PathElement> path;
    StringRef module;
    while (true) {
      if (parseToken(TokenKind::Id, module))
        return failure();
      if (parseToken(TokenKind::Slash))
        break;
      StringRef instance;
      if (parseToken(TokenKind::Id, instance) || parseToken(TokenKind::Colon))
        return failure();
      path.emplace_back(StringAttr::get(context, module),
                        StringAttr::get(context, instance));
    }
    pathAttr = PathAttr::get(context, path);
    moduleAttr = StringAttr::get(context, module);
    if (succeeded(parseToken(TokenKind::Greater))) {
      StringRef ref;
      if (parseToken(TokenKind::Id, ref))
        return failure();
      refAttr = StringAttr::get(context, ref);

      SmallString<64> field;
      while (true) {
        if (succeeded(parseToken(TokenKind::LBrace))) {
          StringRef id;
          if (parseToken(TokenKind::Id, id) || parseToken(TokenKind::RBrace))
            return failure();
          field += '[';
          field += id;
          field += ']';
        } else if (succeeded(parseToken(TokenKind::Period))) {
          StringRef id;
          if (parseToken(TokenKind::Id, id))
            return failure();
          field += '.';
          field += id;
        } else {
          break;
        }
      }
      fieldAttr = StringAttr::get(context, field);
    } else {
      refAttr = StringAttr::get(context, "");
      fieldAttr = StringAttr::get(context, "");
    }
    if (parseToken(TokenKind::Eof))
      return failure();
    return success();
  }

  MLIRContext *context;
  StringRef spelling;
};
} // namespace

ParseResult circt::om::parseBasePath(MLIRContext *context, StringRef spelling,
                                     PathAttr &path) {
  return PathParser{context, spelling}.parseBasePath(path);
}

ParseResult circt::om::parsePath(MLIRContext *context, StringRef spelling,
                                 PathAttr &path, StringAttr &module,
                                 StringAttr &ref, StringAttr &field) {
  return PathParser{context, spelling}.parsePath(path, module, ref, field);
}
