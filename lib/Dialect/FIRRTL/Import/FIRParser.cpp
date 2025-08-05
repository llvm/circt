//===- FIRParser.cpp - .fir to FIRRTL dialect parser ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements a .fir file parser.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "FIRAnnotations.h"
#include "FIRLexer.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Import/FIRAnnotations.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/Timing.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <utility>

using namespace circt;
using namespace firrtl;
using namespace chirrtl;

using llvm::SMLoc;
using llvm::SourceMgr;
using mlir::LocationAttr;

namespace json = llvm::json;

//===----------------------------------------------------------------------===//
// SharedParserConstants
//===----------------------------------------------------------------------===//

namespace {

/// This class refers to immutable values and annotations maintained globally by
/// the parser which can be referred to by any active parser, even those running
/// in parallel.  This is shared by all active parsers.
struct SharedParserConstants {
  SharedParserConstants(MLIRContext *context, FIRParserOptions options)
      : context(context), options(options),
        emptyArrayAttr(ArrayAttr::get(context, {})),
        loIdentifier(StringAttr::get(context, "lo")),
        hiIdentifier(StringAttr::get(context, "hi")),
        amountIdentifier(StringAttr::get(context, "amount")),
        placeholderInnerRef(
            hw::InnerRefAttr::get(StringAttr::get(context, "module"),
                                  StringAttr::get(context, "placeholder"))) {}

  /// The context we're parsing into.
  MLIRContext *const context;

  // Options that control the behavior of the parser.
  const FIRParserOptions options;

  /// A map from identifiers to type aliases.
  llvm::StringMap<FIRRTLType> aliasMap;

  /// A map from identifiers to class ops.
  llvm::DenseMap<StringRef, ClassLike> classMap;

  /// An empty array attribute.
  const ArrayAttr emptyArrayAttr;

  /// Cached identifiers used in primitives.
  const StringAttr loIdentifier, hiIdentifier, amountIdentifier;

  /// Cached placeholder inner-ref used until fixed up.
  const hw::InnerRefAttr placeholderInnerRef;

private:
  SharedParserConstants(const SharedParserConstants &) = delete;
  void operator=(const SharedParserConstants &) = delete;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// FIRParser
//===----------------------------------------------------------------------===//

namespace {
/// This class implements logic common to all levels of the parser, including
/// things like types and helper logic.
struct FIRParser {
  FIRParser(SharedParserConstants &constants, FIRLexer &lexer,
            FIRVersion version)
      : version(version), constants(constants), lexer(lexer),
        locatorFilenameCache(constants.loIdentifier /*arbitrary non-null id*/) {
  }

  // Helper methods to get stuff from the shared parser constants.
  SharedParserConstants &getConstants() const { return constants; }
  MLIRContext *getContext() const { return constants.context; }

  FIRLexer &getLexer() { return lexer; }

  /// Return the indentation level of the specified token.
  std::optional<unsigned> getIndentation() const {
    return lexer.getIndentation(getToken());
  }

  /// Return the current token the parser is inspecting.
  const FIRToken &getToken() const { return lexer.getToken(); }
  StringRef getTokenSpelling() const { return getToken().getSpelling(); }

  //===--------------------------------------------------------------------===//
  // Error Handling
  //===--------------------------------------------------------------------===//

  /// Emit an error and return failure.
  InFlightDiagnostic emitError(const Twine &message = {}) {
    return emitError(getToken().getLoc(), message);
  }
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});

  /// Emit a warning.
  InFlightDiagnostic emitWarning(const Twine &message = {}) {
    return emitWarning(getToken().getLoc(), message);
  }

  InFlightDiagnostic emitWarning(SMLoc loc, const Twine &message = {});

  //===--------------------------------------------------------------------===//
  // Location Handling
  //===--------------------------------------------------------------------===//

  class LocWithInfo;

  /// Encode the specified source location information into an attribute for
  /// attachment to the IR.
  Location translateLocation(llvm::SMLoc loc) {
    return lexer.translateLocation(loc);
  }

  /// Parse an @info marker if present.  If so, fill in the specified Location,
  /// if not, ignore it.
  ParseResult parseOptionalInfoLocator(LocationAttr &result);

  /// Parse an optional name that may appear in Stop, Printf, or Verification
  /// statements.
  ParseResult parseOptionalName(StringAttr &name);

  //===--------------------------------------------------------------------===//
  // Version and Feature Checking
  //===--------------------------------------------------------------------===//

  ParseResult requireFeature(FIRVersion minimum, StringRef feature) {
    return requireFeature(minimum, feature, getToken().getLoc());
  }

  ParseResult requireFeature(FIRVersion minimum, StringRef feature, SMLoc loc) {
    if (version < minimum)
      return emitError(loc)
             << feature << " are a FIRRTL " << minimum
             << "+ feature, but the specified FIRRTL version was " << version;
    return success();
  }

  ParseResult removedFeature(FIRVersion removedVersion, StringRef feature) {
    return removedFeature(removedVersion, feature, getToken().getLoc());
  }

  ParseResult removedFeature(FIRVersion removedVersion, StringRef feature,
                             SMLoc loc) {
    if (version >= removedVersion)
      return emitError(loc)
             << feature << " were removed in FIRRTL " << removedVersion
             << ", but the specified FIRRTL version was " << version;
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Annotation Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a non-standard inline Annotation JSON blob if present.  This uses
  /// the info-like encoding of %[<JSON Blob>].
  ParseResult parseOptionalAnnotations(SMLoc &loc, StringRef &result);

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  /// If the current token has the specified kind, consume it and return true.
  /// If not, return false.
  bool consumeIf(FIRToken::Kind kind) {
    if (getToken().isNot(kind))
      return false;
    consumeToken(kind);
    return true;
  }

  /// Advance the current lexer onto the next token.
  ///
  /// This returns the consumed token.
  FIRToken consumeToken() {
    FIRToken consumedToken = getToken();
    assert(consumedToken.isNot(FIRToken::eof, FIRToken::error) &&
           "shouldn't advance past EOF or errors");
    lexer.lexToken();
    return consumedToken;
  }

  /// Advance the current lexer onto the next token, asserting what the expected
  /// current token is.  This is preferred to the above method because it leads
  /// to more self-documenting code with better checking.
  ///
  /// This returns the consumed token.
  FIRToken consumeToken(FIRToken::Kind kind) {
    FIRToken consumedToken = getToken();
    assert(consumedToken.is(kind) && "consumed an unexpected token");
    consumeToken();
    return consumedToken;
  }

  /// Capture the current token's spelling into the specified value.  This
  /// always succeeds.
  ParseResult parseGetSpelling(StringRef &spelling) {
    spelling = getTokenSpelling();
    return success();
  }

  /// Consume the specified token if present and return success.  On failure,
  /// output a diagnostic and return failure.
  ParseResult parseToken(FIRToken::Kind expectedToken, const Twine &message);

  /// Parse a comma-separated list of elements, terminated with an arbitrary
  /// token.
  ParseResult parseListUntil(FIRToken::Kind rightToken,
                             const std::function<ParseResult()> &parseElement);

  //===--------------------------------------------------------------------===//
  // Common Parser Rules
  //===--------------------------------------------------------------------===//

  /// Parse 'intLit' into the specified value.
  ParseResult parseIntLit(APInt &result, const Twine &message);
  ParseResult parseIntLit(int64_t &result, const Twine &message);
  ParseResult parseIntLit(int32_t &result, const Twine &message);

  // Parse 'verLit' into specified value
  ParseResult parseVersionLit(const Twine &message);

  // Parse ('<' intLit '>')? setting result to -1 if not present.
  template <typename T>
  ParseResult parseOptionalWidth(T &result);

  // Parse the 'id' grammar, which is an identifier or an allowed keyword.
  ParseResult parseId(StringRef &result, const Twine &message);
  ParseResult parseId(StringAttr &result, const Twine &message);
  ParseResult parseFieldId(StringRef &result, const Twine &message);
  ParseResult parseFieldIdSeq(SmallVectorImpl<StringRef> &result,
                              const Twine &message);
  ParseResult parseEnumType(FIRRTLType &result);
  ParseResult parseListType(FIRRTLType &result);
  ParseResult parseType(FIRRTLType &result, const Twine &message);
  // Parse a property type specifically.
  ParseResult parsePropertyType(PropertyType &result, const Twine &message);

  ParseResult parseRUW(RUWBehavior &result);
  ParseResult parseOptionalRUW(RUWBehavior &result);

  ParseResult parseParameter(StringAttr &resultName, Attribute &resultValue,
                             SMLoc &resultLoc, bool allowAggregates = false);
  ParseResult parseParameterValue(Attribute &resultValue,
                                  bool allowAggregates = false);

  /// The version of FIRRTL to use for this parser.
  FIRVersion version;

private:
  FIRParser(const FIRParser &) = delete;
  void operator=(const FIRParser &) = delete;

  /// FIRParser is subclassed and reinstantiated.  Do not add additional
  /// non-trivial state here, add it to SharedParserConstants.
  SharedParserConstants &constants;
  FIRLexer &lexer;

  /// This is a single-entry cache for filenames in locators.
  StringAttr locatorFilenameCache;
  /// This is a single-entry cache for FileLineCol locations.
  FileLineColLoc fileLineColLocCache;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Error Handling
//===----------------------------------------------------------------------===//

InFlightDiagnostic FIRParser::emitError(SMLoc loc, const Twine &message) {
  auto diag = mlir::emitError(translateLocation(loc), message);

  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (getToken().is(FIRToken::error))
    diag.abandon();
  return diag;
}

InFlightDiagnostic FIRParser::emitWarning(SMLoc loc, const Twine &message) {
  return mlir::emitWarning(translateLocation(loc), message);
}

//===----------------------------------------------------------------------===//
// Token Parsing
//===----------------------------------------------------------------------===//

/// Consume the specified token if present and return success.  On failure,
/// output a diagnostic and return failure.
ParseResult FIRParser::parseToken(FIRToken::Kind expectedToken,
                                  const Twine &message) {
  if (consumeIf(expectedToken))
    return success();
  return emitError(message);
}

/// Parse a comma-separated list of zero or more elements, terminated with an
/// arbitrary token.
ParseResult
FIRParser::parseListUntil(FIRToken::Kind rightToken,
                          const std::function<ParseResult()> &parseElement) {
  if (consumeIf(rightToken))
    return success();

  if (parseElement())
    return failure();

  while (consumeIf(FIRToken::comma)) {
    if (parseElement())
      return failure();
  }

  if (parseToken(rightToken, "expected ','"))
    return failure();

  return success();
}

//===--------------------------------------------------------------------===//
// Location Processing
//===--------------------------------------------------------------------===//

/// This helper class is used to handle Info records, which specify higher level
/// symbolic source location, that may be missing from the file.  If the higher
/// level source information is missing, we fall back to the location in the
/// .fir file.
class FIRParser::LocWithInfo {
public:
  explicit LocWithInfo(SMLoc firLoc, FIRParser *parser)
      : parser(parser), firLoc(firLoc) {}

  SMLoc getFIRLoc() const { return firLoc; }

  Location getLoc() {
    if (infoLoc)
      return *infoLoc;
    auto result = parser->translateLocation(firLoc);
    infoLoc = result;
    return result;
  }

  /// Parse an @info marker if present and update our location.
  ParseResult parseOptionalInfo() {
    LocationAttr loc;
    if (failed(parser->parseOptionalInfoLocator(loc)))
      return failure();
    if (loc) {
      using ILH = FIRParserOptions::InfoLocHandling;
      switch (parser->constants.options.infoLocatorHandling) {
      case ILH::IgnoreInfo:
        assert(0 && "Should not return info locations if ignoring");
        break;
      case ILH::PreferInfo:
        infoLoc = loc;
        break;
      case ILH::FusedInfo:
        infoLoc = FusedLoc::get(loc.getContext(),
                                {loc, parser->translateLocation(firLoc)});
        break;
      }
    }
    return success();
  }

  /// If we didn't parse an info locator for the specified value, this sets a
  /// default, overriding a fall back to a location in the .fir file.
  void setDefaultLoc(Location loc) {
    if (!infoLoc)
      infoLoc = loc;
  }

private:
  FIRParser *const parser;

  /// This is the designated location in the .fir file for use when there is no
  /// @ info marker.
  SMLoc firLoc;

  /// This is the location specified by the @ marker if present.
  std::optional<Location> infoLoc;
};

/// Parse an @info marker if present.  If so, fill in the specified Location,
/// if not, ignore it.
ParseResult FIRParser::parseOptionalInfoLocator(LocationAttr &result) {
  if (getToken().isNot(FIRToken::fileinfo))
    return success();

  auto loc = getToken().getLoc();

  auto spelling = getTokenSpelling();
  consumeToken(FIRToken::fileinfo);

  auto locationPair = maybeStringToLocation(
      spelling,
      constants.options.infoLocatorHandling ==
          FIRParserOptions::InfoLocHandling::IgnoreInfo,
      locatorFilenameCache, fileLineColLocCache, getContext());

  // If parsing failed, then indicate that a weird info was found.
  if (!locationPair.first) {
    mlir::emitWarning(translateLocation(loc),
                      "ignoring unknown @ info record format");
    return success();
  }

  // If the parsing succeeded, but we are supposed to drop locators, then just
  // return.
  if (locationPair.first && constants.options.infoLocatorHandling ==
                                FIRParserOptions::InfoLocHandling::IgnoreInfo)
    return success();

  // Otherwise, set the location attribute and return.
  result = *locationPair.second;
  return success();
}

/// Parse an optional trailing name that may show up on assert, assume, cover,
/// stop, or printf.
///
/// optional_name ::= ( ':' id )?
ParseResult FIRParser::parseOptionalName(StringAttr &name) {

  if (getToken().isNot(FIRToken::colon)) {
    name = StringAttr::get(getContext(), "");
    return success();
  }

  consumeToken(FIRToken::colon);
  StringRef nameRef;
  if (parseId(nameRef, "expected result name"))
    return failure();

  name = StringAttr::get(getContext(), nameRef);

  return success();
}

//===--------------------------------------------------------------------===//
// Annotation Handling
//===--------------------------------------------------------------------===//

/// Parse a non-standard inline Annotation JSON blob if present.  This uses
/// the info-like encoding of %[<JSON Blob>].
ParseResult FIRParser::parseOptionalAnnotations(SMLoc &loc, StringRef &result) {

  if (getToken().isNot(FIRToken::inlineannotation))
    return success();

  loc = getToken().getLoc();

  result = getTokenSpelling().drop_front(2).drop_back(1);
  consumeToken(FIRToken::inlineannotation);

  return success();
}

//===--------------------------------------------------------------------===//
// Common Parser Rules
//===--------------------------------------------------------------------===//

/// intLit    ::= UnsignedInt
///           ::= SignedInt
///           ::= HexLit
///           ::= OctalLit
///           ::= BinaryLit
/// HexLit    ::= '"' 'h' ( '+' | '-' )? ( HexDigit )+ '"'
/// OctalLit  ::= '"' 'o' ( '+' | '-' )? ( OctalDigit )+ '"'
/// BinaryLit ::= '"' 'b' ( '+' | '-' )? ( BinaryDigit )+ '"'
///
ParseResult FIRParser::parseIntLit(APInt &result, const Twine &message) {
  auto spelling = getTokenSpelling();
  bool isNegative = false;
  switch (getToken().getKind()) {
  case FIRToken::signed_integer:
    isNegative = spelling[0] == '-';
    assert(spelling[0] == '+' || spelling[0] == '-');
    spelling = spelling.drop_front();
    [[fallthrough]];
  case FIRToken::integer:
    if (spelling.getAsInteger(10, result))
      return emitError(message), failure();

    // Make sure that the returned APInt has a zero at the top so clients don't
    // confuse it with a negative number.
    if (result.isNegative())
      result = result.zext(result.getBitWidth() + 1);

    if (isNegative)
      result = -result;

    // If this was parsed as >32 bits, but can be represented in 32 bits,
    // truncate off the extra width.  This is important for extmodules which
    // like parameters to be 32-bits, and insulates us from some arbitraryness
    // in StringRef::getAsInteger.
    if (result.getBitWidth() > 32 && result.getSignificantBits() <= 32)
      result = result.trunc(32);

    consumeToken();
    return success();
  case FIRToken::radix_specified_integer: {
    if (requireFeature({2, 4, 0}, "radix-specified integer literals"))
      return failure();
    if (spelling[0] == '-') {
      isNegative = true;
      spelling = spelling.drop_front();
    }
    unsigned base = llvm::StringSwitch<unsigned>(spelling.take_front(2))
                        .Case("0b", 2)
                        .Case("0o", 8)
                        .Case("0d", 10)
                        .Case("0h", 16);
    spelling = spelling.drop_front(2);
    if (spelling.getAsInteger(base, result))
      return emitError("invalid character in integer literal"), failure();
    if (result.isNegative())
      result = result.zext(result.getBitWidth() + 1);
    if (isNegative)
      result = -result;
    consumeToken();
    return success();
  }
  case FIRToken::string: {
    if (FIRVersion(3, 0, 0) <= version)
      return emitError(
          "String-encoded integer literals are unsupported after FIRRTL 3.0.0");

    // Drop the quotes.
    assert(spelling.front() == '"' && spelling.back() == '"');
    spelling = spelling.drop_back().drop_front();

    // Decode the base.
    unsigned base;
    switch (spelling.empty() ? ' ' : spelling.front()) {
    case 'h':
      base = 16;
      break;
    case 'o':
      base = 8;
      break;
    case 'b':
      base = 2;
      break;
    default:
      return emitError("expected base specifier (h/o/b) in integer literal"),
             failure();
    }
    spelling = spelling.drop_front();

    // Handle the optional sign.
    bool isNegative = false;
    if (!spelling.empty() && spelling.front() == '+')
      spelling = spelling.drop_front();
    else if (!spelling.empty() && spelling.front() == '-') {
      isNegative = true;
      spelling = spelling.drop_front();
    }

    // Parse the digits.
    if (spelling.empty())
      return emitError("expected digits in integer literal"), failure();

    if (spelling.getAsInteger(base, result))
      return emitError("invalid character in integer literal"), failure();

    // We just parsed the positive version of this number.  Make sure it has
    // a zero at the top so clients don't confuse it with a negative number and
    // so the negation (in the case of a negative sign) doesn't overflow.
    if (result.isNegative())
      result = result.zext(result.getBitWidth() + 1);

    if (isNegative)
      result = -result;

    consumeToken(FIRToken::string);
    return success();
  }

  default:
    return emitError("expected integer literal"), failure();
  }
}

ParseResult FIRParser::parseIntLit(int64_t &result, const Twine &message) {
  APInt value;
  auto loc = getToken().getLoc();
  if (parseIntLit(value, message))
    return failure();

  result = (int64_t)value.getLimitedValue(INT64_MAX);
  if (result != value)
    return emitError(loc, "value is too big to handle"), failure();
  return success();
}

ParseResult FIRParser::parseIntLit(int32_t &result, const Twine &message) {
  APInt value;
  auto loc = getToken().getLoc();
  if (parseIntLit(value, message))
    return failure();

  result = (int32_t)value.getLimitedValue(INT32_MAX);
  if (result != value)
    return emitError(loc, "value is too big to handle"), failure();
  return success();
}

/// versionLit    ::= version
/// deconstruct a version literal into parts and returns those.
ParseResult FIRParser::parseVersionLit(const Twine &message) {
  auto spelling = getTokenSpelling();
  if (getToken().getKind() != FIRToken::version)
    return emitError(message), failure();
  // form a.b.c
  auto [a, d] = spelling.split(".");
  auto [b, c] = d.split(".");
  APInt aInt, bInt, cInt;
  if (a.getAsInteger(10, aInt) || b.getAsInteger(10, bInt) ||
      c.getAsInteger(10, cInt))
    return emitError("failed to parse version string"), failure();
  version.major = aInt.getLimitedValue(UINT32_MAX);
  version.minor = bInt.getLimitedValue(UINT32_MAX);
  version.patch = cInt.getLimitedValue(UINT32_MAX);
  if (version.major != aInt || version.minor != bInt || version.patch != cInt)
    return emitError("integers out of range"), failure();
  if (version < minimumFIRVersion)
    return emitError() << "FIRRTL version must be >=" << minimumFIRVersion,
           failure();
  consumeToken(FIRToken::version);
  return success();
}

// optional-width ::= ('<' intLit '>')?
//
// This returns with result equal to -1 if not present.
template <typename T>
ParseResult FIRParser::parseOptionalWidth(T &result) {
  if (!consumeIf(FIRToken::less))
    return result = -1, success();

  // Parse a width specifier if present.
  auto widthLoc = getToken().getLoc();
  if (parseIntLit(result, "expected width") ||
      parseToken(FIRToken::greater, "expected >"))
    return failure();

  if (result < 0)
    return emitError(widthLoc, "invalid width specifier"), failure();

  return success();
}

/// id  ::= Id | keywordAsId
///
/// Parse the 'id' grammar, which is an identifier or an allowed keyword.  On
/// success, this returns the identifier in the result attribute.
ParseResult FIRParser::parseId(StringRef &result, const Twine &message) {
  switch (getToken().getKind()) {
  // The most common case is an identifier.
  case FIRToken::identifier:
  case FIRToken::literal_identifier:
// Otherwise it may be a keyword that we're allowing in an id position.
#define TOK_KEYWORD(spelling) case FIRToken::kw_##spelling:
#include "FIRTokenKinds.def"

    // Yep, this is a valid identifier or literal identifier.  Turn it into an
    // attribute.  If it is a literal identifier, then drop the leading and
    // trailing '`' (backticks).
    if (getToken().getKind() == FIRToken::literal_identifier)
      result = getTokenSpelling().drop_front().drop_back();
    else
      result = getTokenSpelling();
    consumeToken();
    return success();

  default:
    emitError(message);
    return failure();
  }
}

ParseResult FIRParser::parseId(StringAttr &result, const Twine &message) {
  StringRef name;
  if (parseId(name, message))
    return failure();

  result = StringAttr::get(getContext(), name);
  return success();
}

/// fieldId ::= Id
///         ::= RelaxedId
///         ::= UnsignedInt
///         ::= keywordAsId
///
ParseResult FIRParser::parseFieldId(StringRef &result, const Twine &message) {
  // Handle the UnsignedInt case.
  result = getTokenSpelling();
  if (consumeIf(FIRToken::integer))
    return success();

  // FIXME: Handle RelaxedId

  // Otherwise, it must be Id or keywordAsId.
  if (parseId(result, message))
    return failure();

  return success();
}

/// fieldId ::= Id
///         ::= Float
///         ::= version
///         ::= UnsignedInt
///         ::= keywordAsId
///
ParseResult FIRParser::parseFieldIdSeq(SmallVectorImpl<StringRef> &result,
                                       const Twine &message) {
  // Handle the UnsignedInt case.
  StringRef tmp = getTokenSpelling();

  if (consumeIf(FIRToken::integer)) {
    result.push_back(tmp);
    return success();
  }

  if (consumeIf(FIRToken::floatingpoint)) {
    // form a.b
    // Both a and b could have more floating point stuff, but just ignore that
    // for now.
    auto [a, b] = tmp.split(".");
    result.push_back(a);
    result.push_back(b);
    return success();
  }

  if (consumeIf(FIRToken::version)) {
    // form a.b.c
    auto [a, d] = tmp.split(".");
    auto [b, c] = d.split(".");
    result.push_back(a);
    result.push_back(b);
    result.push_back(c);
    return success();
  }

  // Otherwise, it must be Id or keywordAsId.
  if (parseId(tmp, message))
    return failure();
  result.push_back(tmp);
  return success();
}

/// enum-field ::= Id ( '=' int )? ( ':' type )? ;
/// enum-type  ::= '{|' enum-field* '|}'
ParseResult FIRParser::parseEnumType(FIRRTLType &result) {
  if (parseToken(FIRToken::l_brace_bar,
                 "expected leading '{|' in enumeration type"))
    return failure();
  SmallVector<StringAttr> names;
  SmallVector<APInt> values;
  SmallVector<FIRRTLBaseType> types;
  SmallVector<SMLoc> locs;
  if (parseListUntil(FIRToken::r_brace_bar, [&]() -> ParseResult {
        auto fieldLoc = getToken().getLoc();
        locs.push_back(fieldLoc);

        // Parse the name of the tag.
        StringRef nameStr;
        if (parseId(nameStr, "expected valid identifier for enumeration tag"))
          return failure();
        auto name = StringAttr::get(getContext(), nameStr);
        names.push_back(name);

        // Parse the integer value if it exists. If its the first element of the
        // enum, it implicitly has a value of 0, otherwise it has the previous
        // value + 1.
        APInt value;
        if (consumeIf(FIRToken::equal)) {
          if (parseIntLit(value, "expected integer value for enumeration tag"))
            return failure();
          if (value.isNegative())
            return emitError(fieldLoc, "enum tag value must be non-negative");
        } else if (values.empty()) {
          // This is the first enum variant, so it defaults to 0.
          value = APInt(1, 0);
        } else {
          // This value is not specified, so it defaults to the previous value
          // + 1.
          auto &prev = values.back();
          if (prev.isMaxValue())
            value = prev.zext(prev.getBitWidth() + 1);
          else
            value = prev;
          ++value;
        }
        values.push_back(std::move(value));

        // Parse an optional type ascription.
        FIRRTLBaseType type;
        if (consumeIf(FIRToken::colon)) {
          FIRRTLType parsedType;
          if (parseType(parsedType, "expected enumeration type"))
            return failure();
          type = type_dyn_cast<FIRRTLBaseType>(parsedType);
          if (!type)
            return emitError(fieldLoc, "field must be a base type");
        } else {
          // If there is no type specified, default to UInt<0>.
          type = UIntType::get(getContext(), 0);
        }
        types.push_back(type);

        auto r = type.getRecursiveTypeProperties();
        if (!r.isPassive)
          return emitError(fieldLoc) << "enum field " << name << " not passive";
        if (r.containsAnalog)
          return emitError(fieldLoc)
                 << "enum field " << name << " contains analog";
        if (r.hasUninferredWidth)
          return emitError(fieldLoc)
                 << "enum field " << name << " has uninferred width";
        if (r.hasUninferredReset)
          return emitError(fieldLoc)
                 << "enum field " << name << " has uninferred reset";
        return success();
      }))
    return failure();

  // Verify that the names of each variant are unique.
  SmallPtrSet<StringAttr, 4> nameSet;
  for (auto [name, loc] : llvm::zip(names, locs))
    if (!nameSet.insert(name).second)
      return emitError(loc,
                       "duplicate variant name in enum: " + name.getValue());

  // Find the bitwidth of the enum.
  unsigned bitwidth = 0;
  for (auto &value : values)
    bitwidth = std::max(bitwidth, value.getActiveBits());
  auto tagType =
      IntegerType::get(getContext(), bitwidth, IntegerType::Unsigned);

  // Extend all tag values to the same width, and check that they are all
  // unique.
  SmallPtrSet<IntegerAttr, 4> valueSet;
  SmallVector<FEnumType::EnumElement, 4> elements;
  for (auto [name, value, type, loc] : llvm::zip(names, values, types, locs)) {
    auto tagValue = value.zextOrTrunc(bitwidth);
    auto attr = IntegerAttr::get(tagType, tagValue);
    // Verify that the names of each variant are unique.
    if (!valueSet.insert(attr).second)
      return emitError(loc, "duplicate variant value in enum: ") << attr;
    elements.push_back({name, attr, type});
  }

  llvm::sort(elements);
  result = FEnumType::get(getContext(), elements);
  return success();
}

ParseResult FIRParser::parsePropertyType(PropertyType &result,
                                         const Twine &message) {
  FIRRTLType type;
  if (parseType(type, message))
    return failure();
  auto prop = type_dyn_cast<PropertyType>(type);
  if (!prop)
    return emitError("expected property type");
  result = prop;
  return success();
}

/// list-type ::= 'List' '<' type '>'
ParseResult FIRParser::parseListType(FIRRTLType &result) {
  consumeToken(FIRToken::kw_List);

  PropertyType elementType;
  if (parseToken(FIRToken::less, "expected '<' in List type") ||
      parsePropertyType(elementType, "expected List element type") ||
      parseToken(FIRToken::greater, "expected '>' in List type"))
    return failure();

  result = ListType::get(getContext(), elementType);
  return success();
}

/// type ::= 'Clock'
///      ::= 'Reset'
///      ::= 'AsyncReset'
///      ::= 'UInt' optional-width
///      ::= 'SInt' optional-width
///      ::= 'Analog' optional-width
///      ::= {' field* '}'
///      ::= type '[' intLit ']'
///      ::= 'Probe' '<' type '>'
///      ::= 'RWProbe' '<' type '>'
///      ::= 'const' type
///      ::= 'String'
///      ::= list-type
///      ::= id
///
/// field: 'flip'? fieldId ':' type
///
// NOLINTNEXTLINE(misc-no-recursion)
ParseResult FIRParser::parseType(FIRRTLType &result, const Twine &message) {
  switch (getToken().getKind()) {
  default:
    return emitError(message), failure();

  case FIRToken::kw_Clock:
    consumeToken(FIRToken::kw_Clock);
    result = ClockType::get(getContext());
    break;

  case FIRToken::kw_Inst: {
    if (requireFeature(missingSpecFIRVersion, "Inst types"))
      return failure();

    consumeToken(FIRToken::kw_Inst);
    if (parseToken(FIRToken::less, "expected < in Inst type"))
      return failure();

    auto loc = getToken().getLoc();
    StringRef id;
    if (parseId(id, "expected class name in Inst type"))
      return failure();

    // Look up the class that is being referenced.
    const auto &classMap = getConstants().classMap;
    auto lookup = classMap.find(id);
    if (lookup == classMap.end())
      return emitError(loc) << "unknown class '" << id << "'";

    auto classOp = lookup->second;

    if (parseToken(FIRToken::greater, "expected > in Inst type"))
      return failure();

    result = classOp.getInstanceType();
    break;
  }

  case FIRToken::kw_AnyRef: {
    if (requireFeature(missingSpecFIRVersion, "AnyRef types"))
      return failure();

    consumeToken(FIRToken::kw_AnyRef);
    result = AnyRefType::get(getContext());
    break;
  }

  case FIRToken::kw_Reset:
    consumeToken(FIRToken::kw_Reset);
    result = ResetType::get(getContext());
    break;

  case FIRToken::kw_AsyncReset:
    consumeToken(FIRToken::kw_AsyncReset);
    result = AsyncResetType::get(getContext());
    break;

  case FIRToken::kw_UInt:
  case FIRToken::kw_SInt:
  case FIRToken::kw_Analog: {
    auto kind = getToken().getKind();
    consumeToken();

    // Parse a width specifier if present.
    int32_t width;
    if (parseOptionalWidth(width))
      return failure();

    if (kind == FIRToken::kw_SInt)
      result = SIntType::get(getContext(), width);
    else if (kind == FIRToken::kw_UInt)
      result = UIntType::get(getContext(), width);
    else {
      assert(kind == FIRToken::kw_Analog);
      result = AnalogType::get(getContext(), width);
    }
    break;
  }

  case FIRToken::kw_Probe:
  case FIRToken::kw_RWProbe: {
    auto kind = getToken().getKind();
    auto loc = getToken().getLoc();
    consumeToken();

    // Inner Type
    FIRRTLType type;
    if (parseToken(FIRToken::less, "expected '<' in reference type") ||
        parseType(type, "expected probe data type"))
      return failure();

    SmallVector<StringRef> layers;
    if (consumeIf(FIRToken::comma)) {
      if (requireFeature({4, 0, 0}, "colored probes"))
        return failure();
      // Probe Color
      do {
        StringRef layer;
        loc = getToken().getLoc();
        if (parseId(layer, "expected layer name"))
          return failure();
        layers.push_back(layer);
      } while (consumeIf(FIRToken::period));
    }

    if (!consumeIf(FIRToken::greater))
      return emitError(loc, "expected '>' to end reference type");

    bool forceable = kind == FIRToken::kw_RWProbe;

    auto innerType = type_dyn_cast<FIRRTLBaseType>(type);
    if (!innerType)
      return emitError(loc, "invalid probe inner type, must be base-type");

    if (!innerType.isPassive())
      return emitError(loc, "probe inner type must be passive");

    if (forceable && innerType.containsConst())
      return emitError(loc, "rwprobe cannot contain const");

    SymbolRefAttr layer;
    if (!layers.empty()) {
      auto nestedLayers =
          llvm::map_range(ArrayRef(layers).drop_front(), [&](StringRef a) {
            return FlatSymbolRefAttr::get(getContext(), a);
          });
      layer = SymbolRefAttr::get(getContext(), layers.front(),
                                 llvm::to_vector(nestedLayers));
    }

    result = RefType::get(innerType, forceable, layer);
    break;
  }

  case FIRToken::l_brace: {
    consumeToken(FIRToken::l_brace);

    SmallVector<OpenBundleType::BundleElement, 4> elements;
    SmallPtrSet<StringAttr, 4> nameSet;
    bool bundleCompatible = true;
    if (parseListUntil(FIRToken::r_brace, [&]() -> ParseResult {
          bool isFlipped = consumeIf(FIRToken::kw_flip);

          auto loc = getToken().getLoc();
          StringRef fieldNameStr;
          if (parseFieldId(fieldNameStr, "expected bundle field name") ||
              parseToken(FIRToken::colon, "expected ':' in bundle"))
            return failure();
          auto fieldName = StringAttr::get(getContext(), fieldNameStr);

          // Verify that the names of each field are unique.
          if (!nameSet.insert(fieldName).second)
            return emitError(loc, "duplicate field name in bundle: " +
                                      fieldName.getValue());

          FIRRTLType type;
          if (parseType(type, "expected bundle field type"))
            return failure();

          elements.push_back({fieldName, isFlipped, type});
          bundleCompatible &= isa<BundleType::ElementType>(type);

          return success();
        }))
      return failure();

    // Try to emit base-only bundle.
    if (bundleCompatible) {
      auto bundleElements = llvm::map_range(elements, [](auto element) {
        return BundleType::BundleElement{
            element.name, element.isFlip,
            cast<BundleType::ElementType>(element.type)};
      });
      result = BundleType::get(getContext(), llvm::to_vector(bundleElements));
    } else
      result = OpenBundleType::get(getContext(), elements);
    break;
  }

  case FIRToken::l_brace_bar: {
    if (parseEnumType(result))
      return failure();
    break;
  }

  case FIRToken::identifier: {
    StringRef id;
    auto loc = getToken().getLoc();
    if (parseId(id, "expected a type alias name"))
      return failure();
    auto it = constants.aliasMap.find(id);
    if (it == constants.aliasMap.end()) {
      emitError(loc) << "type identifier `" << id << "` is not declared";
      return failure();
    }
    result = it->second;
    break;
  }

  case FIRToken::kw_const: {
    consumeToken(FIRToken::kw_const);
    auto nextToken = getToken();
    auto loc = nextToken.getLoc();

    // Guard against multiple 'const' specifications
    if (nextToken.is(FIRToken::kw_const))
      return emitError(loc, "'const' can only be specified once on a type");

    if (failed(parseType(result, message)))
      return failure();

    auto baseType = type_dyn_cast<FIRRTLBaseType>(result);
    if (!baseType)
      return emitError(loc, "only hardware types can be 'const'");

    result = baseType.getConstType(true);
    return success();
  }

  case FIRToken::kw_String:
    if (requireFeature({3, 1, 0}, "Strings"))
      return failure();
    consumeToken(FIRToken::kw_String);
    result = StringType::get(getContext());
    break;
  case FIRToken::kw_Integer:
    if (requireFeature({3, 1, 0}, "Integers"))
      return failure();
    consumeToken(FIRToken::kw_Integer);
    result = FIntegerType::get(getContext());
    break;
  case FIRToken::kw_Bool:
    if (requireFeature(missingSpecFIRVersion, "Bools"))
      return failure();
    consumeToken(FIRToken::kw_Bool);
    result = BoolType::get(getContext());
    break;
  case FIRToken::kw_Double:
    if (requireFeature(missingSpecFIRVersion, "Doubles"))
      return failure();
    consumeToken(FIRToken::kw_Double);
    result = DoubleType::get(getContext());
    break;
  case FIRToken::kw_Path:
    if (requireFeature(missingSpecFIRVersion, "Paths"))
      return failure();
    consumeToken(FIRToken::kw_Path);
    result = PathType::get(getContext());
    break;
  case FIRToken::kw_List:
    if (requireFeature({4, 0, 0}, "Lists") || parseListType(result))
      return failure();
    break;
  }

  // Handle postfix vector sizes.
  while (consumeIf(FIRToken::l_square)) {
    auto sizeLoc = getToken().getLoc();
    int64_t size;
    if (parseIntLit(size, "expected width") ||
        parseToken(FIRToken::r_square, "expected ]"))
      return failure();

    if (size < 0)
      return emitError(sizeLoc, "invalid size specifier"), failure();

    auto baseType = type_dyn_cast<FIRRTLBaseType>(result);
    if (baseType)
      result = FVectorType::get(baseType, size);
    else
      result = OpenVectorType::get(result, size);
  }

  return success();
}

/// ruw ::= 'old' | 'new' | 'undefined'
ParseResult FIRParser::parseRUW(RUWBehavior &result) {
  switch (getToken().getKind()) {

  case FIRToken::kw_old:
    result = RUWBehavior::Old;
    consumeToken(FIRToken::kw_old);
    break;
  case FIRToken::kw_new:
    result = RUWBehavior::New;
    consumeToken(FIRToken::kw_new);
    break;
  case FIRToken::kw_undefined:
    result = RUWBehavior::Undefined;
    consumeToken(FIRToken::kw_undefined);
    break;
  default:
    return failure();
  }

  return success();
}

/// ruw ::= 'old' | 'new' | 'undefined'
ParseResult FIRParser::parseOptionalRUW(RUWBehavior &result) {
  switch (getToken().getKind()) {
  default:
    break;

  case FIRToken::kw_old:
    result = RUWBehavior::Old;
    consumeToken(FIRToken::kw_old);
    break;
  case FIRToken::kw_new:
    result = RUWBehavior::New;
    consumeToken(FIRToken::kw_new);
    break;
  case FIRToken::kw_undefined:
    result = RUWBehavior::Undefined;
    consumeToken(FIRToken::kw_undefined);
    break;
  }

  return success();
}

/// param ::= id '=' param-value
ParseResult FIRParser::parseParameter(StringAttr &resultName,
                                      Attribute &resultValue, SMLoc &resultLoc,
                                      bool allowAggregates) {
  auto loc = getToken().getLoc();

  // Parse the name of the parameter.
  StringRef name;
  if (parseId(name, "expected parameter name") ||
      parseToken(FIRToken::equal, "expected '=' in parameter"))
    return failure();

  // Parse the value of the parameter.
  Attribute value;
  if (parseParameterValue(value, allowAggregates))
    return failure();

  resultName = StringAttr::get(getContext(), name);
  resultValue = value;
  resultLoc = loc;
  return success();
}

/// param-value ::= intLit
///             ::= StringLit
///             ::= floatingpoint
///             ::= VerbatimStringLit
///             ::= '[' (param-value)','* ']'   (if allowAggregates)
///             ::= '{' (id '=' param)','* '}'  (if allowAggregates)
ParseResult FIRParser::parseParameterValue(Attribute &value,
                                           bool allowAggregates) {
  mlir::Builder builder(getContext());
  switch (getToken().getKind()) {

  // param-value ::= intLit
  case FIRToken::integer:
  case FIRToken::signed_integer: {
    APInt result;
    if (parseIntLit(result, "invalid integer parameter"))
      return failure();

    // If the integer parameter is less than 32-bits, sign extend this to a
    // 32-bit value.  This needs to eventually emit as a 32-bit value in
    // Verilog and we want to get the size correct immediately.
    if (result.getBitWidth() < 32)
      result = result.sext(32);

    value = builder.getIntegerAttr(
        builder.getIntegerType(result.getBitWidth(), result.isSignBitSet()),
        result);
    return success();
  }

  // param-value ::= StringLit
  case FIRToken::string: {
    // Drop the double quotes and unescape.
    value = builder.getStringAttr(getToken().getStringValue());
    consumeToken(FIRToken::string);
    return success();
  }

  // param-value ::= VerbatimStringLit
  case FIRToken::verbatim_string: {
    // Drop the single quotes and unescape the ones inside.
    auto text = builder.getStringAttr(getToken().getVerbatimStringValue());
    value = hw::ParamVerbatimAttr::get(text);
    consumeToken(FIRToken::verbatim_string);
    return success();
  }

  // param-value ::= floatingpoint
  case FIRToken::floatingpoint: {
    double v;
    if (!llvm::to_float(getTokenSpelling(), v))
      return emitError("invalid float parameter syntax"), failure();

    value = builder.getF64FloatAttr(v);
    consumeToken(FIRToken::floatingpoint);
    return success();
  }

  // param-value ::= '[' (param)','* ']'
  case FIRToken::l_square: {
    if (!allowAggregates)
      return emitError("expected non-aggregate parameter value");
    consumeToken();

    SmallVector<Attribute> elements;
    auto parseElement = [&] {
      return parseParameterValue(elements.emplace_back(),
                                 /*allowAggregates=*/true);
    };
    if (parseListUntil(FIRToken::r_square, parseElement))
      return failure();

    value = builder.getArrayAttr(elements);
    return success();
  }

  // param-value ::= '{' (id '=' param)','* '}'
  case FIRToken::l_brace: {
    if (!allowAggregates)
      return emitError("expected non-aggregate parameter value");
    consumeToken();

    NamedAttrList fields;
    auto parseField = [&]() -> ParseResult {
      StringAttr fieldName;
      Attribute fieldValue;
      SMLoc fieldLoc;
      if (parseParameter(fieldName, fieldValue, fieldLoc,
                         /*allowAggregates=*/true))
        return failure();
      if (fields.set(fieldName, fieldValue))
        return emitError(fieldLoc)
               << "redefinition of parameter '" << fieldName.getValue() << "'";
      return success();
    };
    if (parseListUntil(FIRToken::r_brace, parseField))
      return failure();

    value = fields.getDictionary(getContext());
    return success();
  }

  default:
    return emitError("expected parameter value");
  }
}

//===----------------------------------------------------------------------===//
// FIRModuleContext
//===----------------------------------------------------------------------===//

// Entries in a symbol table are either an mlir::Value for the operation that
// defines the value or an unbundled ID tracking the index in the
// UnbundledValues list.
using UnbundledID = llvm::PointerEmbeddedInt<unsigned, 31>;
using SymbolValueEntry = llvm::PointerUnion<Value, UnbundledID>;

using ModuleSymbolTable =
    llvm::StringMap<std::pair<SMLoc, SymbolValueEntry>, llvm::BumpPtrAllocator>;
using ModuleSymbolTableEntry = ModuleSymbolTable::MapEntryTy;

using UnbundledValueEntry = SmallVector<std::pair<Attribute, Value>>;
using UnbundledValuesList = std::vector<UnbundledValueEntry>;
namespace {
/// This structure is used to track which entries are added while inside a scope
/// and remove them upon exiting the scope.
struct UnbundledValueRestorer {
  UnbundledValuesList &list;
  size_t startingSize;
  UnbundledValueRestorer(UnbundledValuesList &list) : list(list) {
    startingSize = list.size();
  }
  ~UnbundledValueRestorer() { list.resize(startingSize); }
};
} // namespace

using SubaccessCache = llvm::DenseMap<std::pair<Value, unsigned>, Value>;

namespace {
/// This struct provides context information that is global to the module we're
/// currently parsing into.
struct FIRModuleContext : public FIRParser {
  explicit FIRModuleContext(Block *topLevelBlock,
                            SharedParserConstants &constants, FIRLexer &lexer,
                            FIRVersion version)
      : FIRParser(constants, lexer, version), topLevelBlock(topLevelBlock) {}

  /// Get a cached constant.
  template <typename OpTy = ConstantOp, typename... Args>
  Value getCachedConstant(ImplicitLocOpBuilder &builder, Attribute attr,
                          Type type, Args &&...args) {
    auto &result = constantCache[{attr, type}];
    if (result)
      return result;

    // Make sure to insert constants at the top level of the module to maintain
    // dominance.
    OpBuilder::InsertPoint savedIP;

    // Find the insertion point.
    if (builder.getInsertionBlock() != topLevelBlock) {
      savedIP = builder.saveInsertionPoint();
      auto *block = builder.getInsertionBlock();
      while (true) {
        auto *op = block->getParentOp();
        if (!op || !op->getBlock()) {
          // We are inserting into an unknown region.
          builder.setInsertionPointToEnd(topLevelBlock);
          break;
        }
        if (op->getBlock() == topLevelBlock) {
          builder.setInsertionPoint(op);
          break;
        }
        block = op->getBlock();
      }
    }

    result = OpTy::create(builder, type, std::forward<Args>(args)...);

    if (savedIP.isSet())
      builder.setInsertionPoint(savedIP.getBlock(), savedIP.getPoint());

    return result;
  }

  //===--------------------------------------------------------------------===//
  // SubaccessCache

  /// This returns a reference with the assumption that the caller will fill in
  /// the cached value. We keep track of inserted subaccesses so that we can
  /// remove them when we exit a scope.
  Value &getCachedSubaccess(Value value, unsigned index) {
    auto &result = subaccessCache[{value, index}];
    if (!result) {
      // The outer most block won't be in the map.
      auto it = scopeMap.find(value.getParentBlock());
      if (it != scopeMap.end())
        it->second->scopedSubaccesses.push_back({result, index});
    }
    return result;
  }

  //===--------------------------------------------------------------------===//
  // SymbolTable

  /// Add a symbol entry with the specified name, returning failure if the name
  /// is already defined.
  ParseResult addSymbolEntry(StringRef name, SymbolValueEntry entry, SMLoc loc,
                             bool insertNameIntoGlobalScope = false);
  ParseResult addSymbolEntry(StringRef name, Value value, SMLoc loc,
                             bool insertNameIntoGlobalScope = false) {
    return addSymbolEntry(name, SymbolValueEntry(value), loc,
                          insertNameIntoGlobalScope);
  }

  // Removes a symbol from symbolTable (Workaround since symbolTable is private)
  void removeSymbolEntry(StringRef name);

  /// Resolved a symbol table entry to a value.  Emission of error is optional.
  ParseResult resolveSymbolEntry(Value &result, SymbolValueEntry &entry,
                                 SMLoc loc, bool fatal = true);

  /// Resolved a symbol table entry if it is an expanded bundle e.g. from an
  /// instance.  Emission of error is optional.
  ParseResult resolveSymbolEntry(Value &result, SymbolValueEntry &entry,
                                 StringRef field, SMLoc loc);

  /// Look up the specified name, emitting an error and returning failure if the
  /// name is unknown.
  ParseResult lookupSymbolEntry(SymbolValueEntry &result, StringRef name,
                                SMLoc loc);

  UnbundledValueEntry &getUnbundledEntry(unsigned index) {
    assert(index < unbundledValues.size());
    return unbundledValues[index];
  }

  /// This contains one entry for each value in FIRRTL that is represented as a
  /// bundle type in the FIRRTL spec but for which we represent as an exploded
  /// set of elements in the FIRRTL dialect.
  UnbundledValuesList unbundledValues;

  /// Provide a symbol table scope that automatically pops all the entries off
  /// the symbol table when the scope is exited.
  struct ContextScope {
    friend struct FIRModuleContext;
    ContextScope(FIRModuleContext &moduleContext, Block *block)
        : moduleContext(moduleContext), block(block),
          previousScope(moduleContext.currentScope) {
      moduleContext.currentScope = this;
      moduleContext.scopeMap[block] = this;
    }
    ~ContextScope() {
      // Mark all entries in this scope as being invalid.  We track validity
      // through the SMLoc field instead of deleting entries.
      for (auto *entryPtr : scopedDecls)
        entryPtr->second.first = SMLoc();
      // Erase the scoped subacceses from the cache. If the block is deleted we
      // could resuse the memory, although the chances are quite small.
      for (auto subaccess : scopedSubaccesses)
        moduleContext.subaccessCache.erase(subaccess);
      // Erase this context from the map.
      moduleContext.scopeMap.erase(block);
      // Reset to the previous scope.
      moduleContext.currentScope = previousScope;
    }

  private:
    void operator=(const ContextScope &) = delete;
    ContextScope(const ContextScope &) = delete;

    FIRModuleContext &moduleContext;
    Block *block;
    ContextScope *previousScope;
    std::vector<ModuleSymbolTableEntry *> scopedDecls;
    std::vector<std::pair<Value, unsigned>> scopedSubaccesses;
  };

private:
  /// The top level block in which we insert cached constants.
  Block *topLevelBlock;

  /// The expression-oriented nature of firrtl syntax produces tons of constant
  /// nodes which are obviously redundant.  Instead of literally producing them
  /// in the parser, do an implicit CSE to reduce parse time and silliness in
  /// the resulting IR.
  llvm::DenseMap<std::pair<Attribute, Type>, Value> constantCache;

  /// This symbol table holds the names of ports, wires, and other local decls.
  /// This is scoped because conditional statements introduce subscopes.
  ModuleSymbolTable symbolTable;

  /// This is a cache of subindex and subfield operations so we don't constantly
  /// recreate large chains of them.  This maps a bundle value + index to the
  /// subaccess result.
  SubaccessCache subaccessCache;

  /// This maps a block to related ContextScope.
  DenseMap<Block *, ContextScope *> scopeMap;

  /// If non-null, all new entries added to the symbol table are added to this
  /// list.  This allows us to "pop" the entries by resetting them to null when
  /// scope is exited.
  ContextScope *currentScope = nullptr;
};

} // end anonymous namespace

// Removes a symbol from symbolTable (Workaround since symbolTable is private)
void FIRModuleContext::removeSymbolEntry(StringRef name) {
  symbolTable.erase(name);
}

/// Add a symbol entry with the specified name, returning failure if the name
/// is already defined.
///
/// When 'insertNameIntoGlobalScope' is true, we don't allow the name to be
/// popped.  This is a workaround for (firrtl scala bug) that should eventually
/// be fixed.
ParseResult FIRModuleContext::addSymbolEntry(StringRef name,
                                             SymbolValueEntry entry, SMLoc loc,
                                             bool insertNameIntoGlobalScope) {
  // Do a lookup by trying to do an insertion.  Do so in a way that we can tell
  // if we hit a missing element (SMLoc is null).
  auto [entryIt, inserted] =
      symbolTable.try_emplace(name, SMLoc(), SymbolValueEntry());

  // If insertion failed, the name already exists
  if (!inserted) {
    if (entryIt->second.first.isValid()) {
      // Valid activeSMLoc: active symbol in current scope redeclared
      emitError(loc, "redefinition of name '" + name + "' ")
              .attachNote(translateLocation(entryIt->second.first))
          << "previous definition here.";
    } else {
      // Invalid activeSMLoc: symbol from a completed scope redeclared
      emitError(loc, "redefinition of name '" + name + "' ")
          << "- FIRRTL has flat namespace and requires all "
          << "declarations in a module to have unique names.";
    }
    return failure();
  }

  // If we didn't have a hit, then record the location, and remember that this
  // was new to this scope.
  entryIt->second = {loc, entry};
  if (currentScope && !insertNameIntoGlobalScope)
    currentScope->scopedDecls.push_back(&*entryIt);
  return success();
}

/// Look up the specified name, emitting an error and returning null if the
/// name is unknown.
ParseResult FIRModuleContext::lookupSymbolEntry(SymbolValueEntry &result,
                                                StringRef name, SMLoc loc) {
  auto &entry = symbolTable[name];
  if (!entry.first.isValid())
    return emitError(loc, "use of unknown declaration '" + name + "'");
  result = entry.second;
  assert(result && "name in symbol table without definition");
  return success();
}

ParseResult FIRModuleContext::resolveSymbolEntry(Value &result,
                                                 SymbolValueEntry &entry,
                                                 SMLoc loc, bool fatal) {
  if (!isa<Value>(entry)) {
    if (fatal)
      emitError(loc, "bundle value should only be used from subfield");
    return failure();
  }
  result = cast<Value>(entry);
  return success();
}

ParseResult FIRModuleContext::resolveSymbolEntry(Value &result,
                                                 SymbolValueEntry &entry,
                                                 StringRef fieldName,
                                                 SMLoc loc) {
  if (!isa<UnbundledID>(entry)) {
    emitError(loc, "value should not be used from subfield");
    return failure();
  }

  auto fieldAttr = StringAttr::get(getContext(), fieldName);

  unsigned unbundledId = cast<UnbundledID>(entry) - 1;
  assert(unbundledId < unbundledValues.size());
  UnbundledValueEntry &ubEntry = unbundledValues[unbundledId];
  for (auto elt : ubEntry) {
    if (elt.first == fieldAttr) {
      result = elt.second;
      break;
    }
  }
  if (!result) {
    emitError(loc, "use of invalid field name '")
        << fieldName << "' on bundle value";
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FIRStmtParser
//===----------------------------------------------------------------------===//

namespace {
/// This class is used when building expression nodes for a statement: we need
/// to parse a bunch of expressions and build MLIR operations for them, and then
/// we see the locator that specifies the location for those operations
/// afterward.
///
/// It is wasteful in time and memory to create a bunch of temporary FileLineCol
/// location's that point into the .fir file when they're destined to get
/// overwritten by a location specified by a Locator.  To avoid this, we create
/// all of the operations with a temporary location on them, then remember the
/// [Operation*, SMLoc] pair for the newly created operation.
///
/// At the end of the operation we'll see a Locator (or not).  If we see a
/// locator, we apply it to all the operations we've parsed and we're done.  If
/// not, we lazily create the locators in the .fir file.
struct LazyLocationListener : public OpBuilder::Listener {
  LazyLocationListener(OpBuilder &builder) : builder(builder) {
    assert(builder.getListener() == nullptr);
    builder.setListener(this);
  }

  ~LazyLocationListener() {
    assert(subOps.empty() && "didn't process parsed operations");
    assert(builder.getListener() == this);
    builder.setListener(nullptr);
  }

  void startStatement() {
    assert(!isActive && "Already processing a statement");
    isActive = true;
  }

  /// This is called when done with each statement.  This applies the locations
  /// to each statement.
  void endStatement(FIRParser &parser) {
    assert(isActive && "Not parsing a statement");

    // If we have a symbolic location, apply it to any subOps specified.
    if (infoLoc) {
      for (auto opAndSMLoc : subOps) {
        // Follow user preference to either only use @info locations,
        // or apply a fused location with @info and file loc.
        using ILH = FIRParserOptions::InfoLocHandling;
        switch (parser.getConstants().options.infoLocatorHandling) {
        case ILH::IgnoreInfo:
          // Shouldn't have an infoLoc, but if we do ignore it.
          opAndSMLoc.first->setLoc(parser.translateLocation(opAndSMLoc.second));
          break;
        case ILH::PreferInfo:
          opAndSMLoc.first->setLoc(infoLoc);
          break;
        case ILH::FusedInfo:
          opAndSMLoc.first->setLoc(FusedLoc::get(
              infoLoc.getContext(),
              {infoLoc, parser.translateLocation(opAndSMLoc.second)}));
          break;
        }
      }
    } else {
      // If we don't, translate all the individual SMLoc's to Location objects
      // in the .fir file.
      for (auto opAndSMLoc : subOps)
        opAndSMLoc.first->setLoc(parser.translateLocation(opAndSMLoc.second));
    }

    // Reset our state.
    isActive = false;
    infoLoc = LocationAttr();
    currentSMLoc = SMLoc();
    subOps.clear();
  }

  /// Specify the location to be used for the next operations that are created.
  void setLoc(SMLoc loc) { currentSMLoc = loc; }

  /// When a @Info locator is parsed, this method captures it.
  void setInfoLoc(LocationAttr loc) {
    assert(!infoLoc && "Info location multiply specified");
    infoLoc = loc;
  }

  // Notification handler for when an operation is inserted into the builder.
  /// `op` is the operation that was inserted.
  void notifyOperationInserted(Operation *op,
                               mlir::IRRewriter::InsertPoint) override {
    assert(currentSMLoc != SMLoc() && "No .fir file location specified");
    assert(isActive && "Not parsing a statement");
    subOps.push_back({op, currentSMLoc});
  }

private:
  /// This is set to true while parsing a statement.  It is used for assertions.
  bool isActive = false;

  /// This is the current position in the source file that the next operation
  /// will be parsed into.
  SMLoc currentSMLoc;

  /// This is the @ location attribute for the current statement, or null if
  /// not set.
  LocationAttr infoLoc;

  /// This is the builder we're installed into.
  OpBuilder &builder;

  /// This is the set of operations we've enqueued along with their location in
  /// the source file.
  SmallVector<std::pair<Operation *, SMLoc>, 8> subOps;

  void operator=(const LazyLocationListener &) = delete;
  LazyLocationListener(const LazyLocationListener &) = delete;
};
} // end anonymous namespace

namespace {
/// This class tracks inner-ref users and their intended targets,
/// (presently there must be just one) for post-processing at a point
/// where adding the symbols is safe without risk of races.
struct InnerSymFixups {
  /// Add a fixup to be processed later.
  void add(hw::InnerRefUserOpInterface user, hw::InnerSymTarget target) {
    fixups.push_back({user, target});
  }

  /// Resolve all stored fixups, if any.  Not expected to fail,
  /// as checking should primarily occur during original parsing.
  LogicalResult resolve(hw::InnerSymbolNamespaceCollection &isnc);

private:
  struct Fixup {
    hw::InnerRefUserOpInterface innerRefUser;
    hw::InnerSymTarget target;
  };
  SmallVector<Fixup, 0> fixups;
};
} // end anonymous namespace

LogicalResult
InnerSymFixups::resolve(hw::InnerSymbolNamespaceCollection &isnc) {
  for (auto &f : fixups) {
    auto ref = getInnerRefTo(
        f.target, [&isnc](FModuleLike module) -> hw::InnerSymbolNamespace & {
          return isnc.get(module);
        });
    assert(ref && "unable to resolve inner symbol target");

    // Per-op fixup logic.  Only RWProbeOp's presently.
    auto result =
        TypeSwitch<Operation *, LogicalResult>(f.innerRefUser.getOperation())
            .Case<RWProbeOp>([ref](RWProbeOp op) {
              op.setTargetAttr(ref);
              return success();
            })
            .Default([](auto *op) {
              return op->emitError("unknown inner-ref user requiring fixup");
            });
    if (failed(result))
      return failure();
  }
  return success();
}

namespace {
/// This class implements logic and state for parsing statements, suites, and
/// similar module body constructs.
struct FIRStmtParser : public FIRParser {
  explicit FIRStmtParser(Block &blockToInsertInto,
                         FIRModuleContext &moduleContext,
                         InnerSymFixups &innerSymFixups,
                         const SymbolTable &circuitSymTbl, FIRVersion version,
                         SymbolRefAttr layerSym = {})
      : FIRParser(moduleContext.getConstants(), moduleContext.getLexer(),
                  version),
        builder(UnknownLoc::get(getContext()), getContext()),
        locationProcessor(this->builder), moduleContext(moduleContext),
        innerSymFixups(innerSymFixups), layerSym(layerSym),
        circuitSymTbl(circuitSymTbl) {
    builder.setInsertionPointToEnd(&blockToInsertInto);
  }

  ParseResult parseSimpleStmt(unsigned stmtIndent);
  ParseResult parseSimpleStmtBlock(unsigned indent);

private:
  ParseResult parseSimpleStmtImpl(unsigned stmtIndent);

  /// Attach invalid values to every element of the value.
  void emitInvalidate(Value val, Flow flow);

  // The FIRRTL specification describes Invalidates as a statement with
  // implicit connect semantics.  The FIRRTL dialect models it as a primitive
  // that returns an "Invalid Value", followed by an explicit connect to make
  // the representation simpler and more consistent.
  void emitInvalidate(Value val) { emitInvalidate(val, foldFlow(val)); }

  /// Parse an @info marker if present and inform locationProcessor about it.
  ParseResult parseOptionalInfo() {
    LocationAttr loc;
    if (failed(parseOptionalInfoLocator(loc)))
      return failure();
    locationProcessor.setInfoLoc(loc);
    return success();
  }

  // Exp Parsing
  ParseResult parseExpImpl(Value &result, const Twine &message,
                           bool isLeadingStmt);
  ParseResult parseExp(Value &result, const Twine &message) {
    return parseExpImpl(result, message, /*isLeadingStmt:*/ false);
  }
  ParseResult parseExpLeadingStmt(Value &result, const Twine &message) {
    return parseExpImpl(result, message, /*isLeadingStmt:*/ true);
  }
  ParseResult parseEnumExp(Value &result);
  ParseResult parsePathExp(Value &result);
  ParseResult parseRefExp(Value &result, const Twine &message);
  ParseResult parseStaticRefExp(Value &result, const Twine &message);
  ParseResult parseRWProbeStaticRefExp(FieldRef &refResult, Type &type,
                                       const Twine &message);

  // Generic intrinsic parsing.
  ParseResult parseIntrinsic(Value &result, bool isStatement);
  ParseResult parseIntrinsicStmt() {
    Value unused;
    return parseIntrinsic(unused, /*isStatement=*/true);
  }
  ParseResult parseIntrinsicExp(Value &result) {
    return parseIntrinsic(result, /*isStatement=*/false);
  }
  ParseResult parseOptionalParams(ArrayAttr &resultParameters);

  template <typename subop>
  FailureOr<Value> emitCachedSubAccess(Value base, unsigned indexNo, SMLoc loc);
  ParseResult parseOptionalExpPostscript(Value &result,
                                         bool allowDynamic = true);
  ParseResult parsePostFixFieldId(Value &result);
  ParseResult parsePostFixIntSubscript(Value &result);
  ParseResult parsePostFixDynamicSubscript(Value &result);
  ParseResult parseIntegerLiteralExp(Value &result);
  ParseResult parseListExp(Value &result);
  ParseResult parseListConcatExp(Value &result);
  ParseResult parseCatExp(Value &result);

  template <typename T, size_t M, size_t N, size_t... Ms, size_t... Ns>
  ParseResult parsePrim(std::index_sequence<Ms...>, std::index_sequence<Ns...>,
                        Value &result) {
    auto loc = getToken().getLoc();
    locationProcessor.setLoc(loc);
    consumeToken();

    auto vals = std::array<Value, M>();
    auto ints = std::array<int64_t, N>();

    // Parse all the values.
    bool first = true;
    for (size_t i = 0; i < M; ++i) {
      if (!first)
        if (parseToken(FIRToken::comma, "expected ','"))
          return failure();
      if (parseExp(vals[i], "expected expression in primitive operand"))
        return failure();
      first = false;
    }

    // Parse all the attributes.
    for (size_t i = 0; i < N; ++i) {
      if (!first)
        if (parseToken(FIRToken::comma, "expected ','"))
          return failure();
      if (parseIntLit(ints[i], "expected integer in primitive operand"))
        return failure();
      first = false;
    }

    if (parseToken(FIRToken::r_paren, "expected ')'"))
      return failure();

    // Infer the type.
    auto type = T::inferReturnType(cast<FIRRTLType>(vals[Ms].getType())...,
                                   ints[Ns]..., {});
    if (!type) {
      // Only call translateLocation on an error case, it is expensive.
      T::inferReturnType(cast<FIRRTLType>(vals[Ms].getType())..., ints[Ns]...,
                         translateLocation(loc));
      return failure();
    }

    // Create the operation.
    auto op = T::create(builder, type, vals[Ms]..., ints[Ns]...);
    result = op.getResult();
    return success();
  }

  template <typename T, unsigned M, unsigned N>
  ParseResult parsePrimExp(Value &result) {
    auto ms = std::make_index_sequence<M>();
    auto ns = std::make_index_sequence<N>();
    return parsePrim<T, M, N>(ms, ns, result);
  }

  std::optional<ParseResult> parseExpWithLeadingKeyword(FIRToken keyword);

  // Stmt Parsing
  ParseResult parseSubBlock(Block &blockToInsertInto, unsigned indent,
                            SymbolRefAttr layerSym);
  ParseResult parseAttach();
  ParseResult parseMemPort(MemDirAttr direction);

  // Parse a format string and build operations for FIRRTL "special"
  // substitutions. Set `formatStringResult` to the validated format string and
  // `operands` to the list of actual operands.
  ParseResult parseFormatString(SMLoc formatStringLoc, StringRef formatString,
                                ArrayRef<Value> specOperands,
                                StringAttr &formatStringResult,
                                SmallVectorImpl<Value> &operands);
  ParseResult parsePrintf();
  ParseResult parseFPrintf();
  ParseResult parseFFlush();
  ParseResult parseSkip();
  ParseResult parseStop();
  ParseResult parseAssert();
  ParseResult parseAssume();
  ParseResult parseCover();
  ParseResult parseWhen(unsigned whenIndent);
  ParseResult parseMatch(unsigned matchIndent);
  ParseResult parseRefDefine();
  ParseResult parseRefForce();
  ParseResult parseRefForceInitial();
  ParseResult parseRefRelease();
  ParseResult parseRefReleaseInitial();
  ParseResult parseRefRead(Value &result);
  ParseResult parseProbe(Value &result);
  ParseResult parsePropAssign();
  ParseResult parseRWProbe(Value &result);
  ParseResult parseLeadingExpStmt(Value lhs);
  ParseResult parseConnect();
  ParseResult parseInvalidate();
  ParseResult parseLayerBlockOrGroup(unsigned indent);

  // Declarations
  ParseResult parseInstance();
  ParseResult parseInstanceChoice();
  ParseResult parseObject();
  ParseResult parseCombMem();
  ParseResult parseSeqMem();
  ParseResult parseMem(unsigned memIndent);
  ParseResult parseNode();
  ParseResult parseWire();
  ParseResult parseRegister(unsigned regIndent);
  ParseResult parseRegisterWithReset();
  ParseResult parseContract(unsigned blockIndent);

  // Helper to fetch a module referenced by an instance-like statement.
  FModuleLike getReferencedModule(SMLoc loc, StringRef moduleName);

  // The builder to build into.
  ImplicitLocOpBuilder builder;
  LazyLocationListener locationProcessor;

  // Extra information maintained across a module.
  FIRModuleContext &moduleContext;

  /// Inner symbol users to fixup after parsing.
  InnerSymFixups &innerSymFixups;

  // An optional symbol that contains the current layer block that we are in.
  // This is used to construct a nested symbol for a layer block operation.
  SymbolRefAttr layerSym;

  const SymbolTable &circuitSymTbl;
};

} // end anonymous namespace

/// Attach invalid values to every element of the value.
// NOLINTNEXTLINE(misc-no-recursion)
void FIRStmtParser::emitInvalidate(Value val, Flow flow) {
  auto tpe = type_dyn_cast<FIRRTLBaseType>(val.getType());
  // Invalidate does nothing for non-base types.
  // When aggregates-of-refs are supported, instead check 'containsReference'
  // below.
  if (!tpe)
    return;

  auto props = tpe.getRecursiveTypeProperties();
  if (props.isPassive && !props.containsAnalog) {
    if (flow == Flow::Source)
      return;
    emitConnect(builder, val, InvalidValueOp::create(builder, tpe));
    return;
  }

  // Recurse until we hit passive leaves.  Connect any leaves which have sink or
  // duplex flow.
  //
  // TODO: This is very similar to connect expansion in the LowerTypes pass
  // works.  Find a way to unify this with methods common to LowerTypes or to
  // have LowerTypes to the actual work here, e.g., emitting a partial connect
  // to only the leaf sources.
  TypeSwitch<FIRRTLType>(tpe)
      .Case<BundleType>([&](auto tpe) {
        for (size_t i = 0, e = tpe.getNumElements(); i < e; ++i) {
          auto &subfield = moduleContext.getCachedSubaccess(val, i);
          if (!subfield) {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointAfterValue(val);
            subfield = SubfieldOp::create(builder, val, i);
          }
          emitInvalidate(subfield,
                         tpe.getElement(i).isFlip ? swapFlow(flow) : flow);
        }
      })
      .Case<FVectorType>([&](auto tpe) {
        auto tpex = tpe.getElementType();
        for (size_t i = 0, e = tpe.getNumElements(); i != e; ++i) {
          auto &subindex = moduleContext.getCachedSubaccess(val, i);
          if (!subindex) {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointAfterValue(val);
            subindex = SubindexOp::create(builder, tpex, val, i);
          }
          emitInvalidate(subindex, flow);
        }
      });
}

//===-------------------------------
// FIRStmtParser Expression Parsing.

/// Parse the 'exp' grammar, returning all of the suboperations in the
/// specified vector, and the ultimate SSA value in value.
///
///  exp ::= id    // Ref
///      ::= prim
///      ::= integer-literal-exp
///      ::= enum-exp
///      ::= list-exp
///      ::= 'String(' stringLit ')'
///      ::= exp '.' fieldId
///      ::= exp '[' intLit ']'
/// XX   ::= exp '.' DoubleLit // TODO Workaround for #470
///      ::= exp '[' exp ']'
///
///
/// If 'isLeadingStmt' is true, then this is being called to parse the first
/// expression in a statement.  We can handle some weird cases due to this if
/// we end up parsing the whole statement.  In that case we return success, but
/// set the 'result' value to null.
// NOLINTNEXTLINE(misc-no-recursion)
ParseResult FIRStmtParser::parseExpImpl(Value &result, const Twine &message,
                                        bool isLeadingStmt) {
  auto token = getToken();
  auto kind = token.getKind();
  switch (kind) {
  case FIRToken::lp_integer_add:
  case FIRToken::lp_integer_mul:
  case FIRToken::lp_integer_shr:
  case FIRToken::lp_integer_shl:
    if (requireFeature({4, 0, 0}, "Integer arithmetic expressions"))
      return failure();
    break;
  default:
    break;
  }

  switch (kind) {
    // Handle all primitive's.
#define TOK_LPKEYWORD_PRIM(SPELLING, CLASS, NUMOPERANDS, NUMATTRIBUTES,        \
                           VERSION, FEATURE)                                   \
  case FIRToken::lp_##SPELLING:                                                \
    if (requireFeature(VERSION, FEATURE))                                      \
      return failure();                                                        \
    if (parsePrimExp<CLASS, NUMOPERANDS, NUMATTRIBUTES>(result))               \
      return failure();                                                        \
    break;
#include "FIRTokenKinds.def"

  case FIRToken::l_brace_bar:
    if (isLeadingStmt)
      return emitError("unexpected enumeration as start of statement");
    if (parseEnumExp(result))
      return failure();
    break;
  case FIRToken::lp_read:
    if (isLeadingStmt)
      return emitError("unexpected read() as start of statement");
    if (parseRefRead(result))
      return failure();
    break;
  case FIRToken::lp_probe:
    if (isLeadingStmt)
      return emitError("unexpected probe() as start of statement");
    if (parseProbe(result))
      return failure();
    break;
  case FIRToken::lp_rwprobe:
    if (isLeadingStmt)
      return emitError("unexpected rwprobe() as start of statement");
    if (parseRWProbe(result))
      return failure();
    break;

  case FIRToken::kw_UInt:
  case FIRToken::kw_SInt:
    if (parseIntegerLiteralExp(result))
      return failure();
    break;
  case FIRToken::kw_String: {
    if (requireFeature({3, 1, 0}, "Strings"))
      return failure();
    locationProcessor.setLoc(getToken().getLoc());
    consumeToken(FIRToken::kw_String);
    StringRef spelling;
    if (parseToken(FIRToken::l_paren, "expected '(' in String expression") ||
        parseGetSpelling(spelling) ||
        parseToken(FIRToken::string,
                   "expected string literal in String expression") ||
        parseToken(FIRToken::r_paren, "expected ')' in String expression"))
      return failure();
    auto attr = builder.getStringAttr(FIRToken::getStringValue(spelling));
    result = moduleContext.getCachedConstant<StringConstantOp>(
        builder, attr, builder.getType<StringType>(), attr);
    break;
  }
  case FIRToken::kw_Integer: {
    if (requireFeature({3, 1, 0}, "Integers"))
      return failure();
    locationProcessor.setLoc(getToken().getLoc());
    consumeToken(FIRToken::kw_Integer);
    APInt value;
    if (parseToken(FIRToken::l_paren, "expected '(' in Integer expression") ||
        parseIntLit(value, "expected integer literal in Integer expression") ||
        parseToken(FIRToken::r_paren, "expected ')' in Integer expression"))
      return failure();
    APSInt apint(value, /*isUnsigned=*/false);
    result = moduleContext.getCachedConstant<FIntegerConstantOp>(
        builder, IntegerAttr::get(getContext(), apint),
        builder.getType<FIntegerType>(), apint);
    break;
  }
  case FIRToken::kw_Bool: {
    if (requireFeature(missingSpecFIRVersion, "Bools"))
      return failure();
    locationProcessor.setLoc(getToken().getLoc());
    consumeToken(FIRToken::kw_Bool);
    if (parseToken(FIRToken::l_paren, "expected '(' in Bool expression"))
      return failure();
    bool value;
    if (consumeIf(FIRToken::kw_true))
      value = true;
    else if (consumeIf(FIRToken::kw_false))
      value = false;
    else
      return emitError("expected true or false in Bool expression");
    if (parseToken(FIRToken::r_paren, "expected ')' in Bool expression"))
      return failure();
    auto attr = builder.getBoolAttr(value);
    result = moduleContext.getCachedConstant<BoolConstantOp>(
        builder, attr, builder.getType<BoolType>(), value);
    break;
  }
  case FIRToken::kw_Double: {
    if (requireFeature(missingSpecFIRVersion, "Doubles"))
      return failure();
    locationProcessor.setLoc(getToken().getLoc());
    consumeToken(FIRToken::kw_Double);
    if (parseToken(FIRToken::l_paren, "expected '(' in Double expression"))
      return failure();
    auto spelling = getTokenSpelling();
    if (parseToken(FIRToken::floatingpoint,
                   "expected floating point in Double expression") ||
        parseToken(FIRToken::r_paren, "expected ')' in Double expression"))
      return failure();
    // NaN, INF, exponent, hex, integer?
    // This uses `strtod` internally, FWIW.  See `man 3 strtod`.
    double d;
    if (!llvm::to_float(spelling, d))
      return emitError("invalid double");
    auto attr = builder.getF64FloatAttr(d);
    result = moduleContext.getCachedConstant<DoubleConstantOp>(
        builder, attr, builder.getType<DoubleType>(), attr);
    break;
  }
  case FIRToken::kw_List: {
    if (requireFeature({4, 0, 0}, "Lists"))
      return failure();
    if (isLeadingStmt)
      return emitError("unexpected List<>() as start of statement");
    if (parseListExp(result))
      return failure();
    break;
  }

  case FIRToken::lp_list_concat: {
    if (isLeadingStmt)
      return emitError("unexpected list_create() as start of statement");
    if (requireFeature({4, 0, 0}, "List concat") || parseListConcatExp(result))
      return failure();
    break;
  }

  case FIRToken::lp_path:
    if (isLeadingStmt)
      return emitError("unexpected path() as start of statement");
    if (requireFeature(missingSpecFIRVersion, "Paths") || parsePathExp(result))
      return failure();
    break;

  case FIRToken::lp_intrinsic:
    if (requireFeature({4, 0, 0}, "generic intrinsics") ||
        parseIntrinsicExp(result))
      return failure();
    break;

  case FIRToken::lp_cat:
    if (parseCatExp(result))
      return failure();
    break;

    // Otherwise there are a bunch of keywords that are treated as identifiers
    // try them.
  case FIRToken::identifier: // exp ::= id
  case FIRToken::literal_identifier:
  default: {
    StringRef name;
    auto loc = getToken().getLoc();
    SymbolValueEntry symtabEntry;
    if (parseId(name, message) ||
        moduleContext.lookupSymbolEntry(symtabEntry, name, loc))
      return failure();

    // If we looked up a normal value, then we're done.
    if (!moduleContext.resolveSymbolEntry(result, symtabEntry, loc, false))
      break;

    assert(isa<UnbundledID>(symtabEntry) && "should be an instance");

    // Otherwise we referred to an implicitly bundled value.  We *must* be in
    // the midst of processing a field ID reference or 'is invalid'.  If not,
    // this is an error.
    if (isLeadingStmt && consumeIf(FIRToken::kw_is)) {
      if (parseToken(FIRToken::kw_invalid, "expected 'invalid'") ||
          parseOptionalInfo())
        return failure();

      locationProcessor.setLoc(loc);
      // Invalidate all of the results of the bundled value.
      unsigned unbundledId = cast<UnbundledID>(symtabEntry) - 1;
      UnbundledValueEntry &ubEntry =
          moduleContext.getUnbundledEntry(unbundledId);
      for (auto elt : ubEntry)
        emitInvalidate(elt.second);

      // Signify that we parsed the whole statement.
      result = Value();
      return success();
    }

    // Handle the normal "instance.x" reference.
    StringRef fieldName;
    if (parseToken(FIRToken::period, "expected '.' in field reference") ||
        parseFieldId(fieldName, "expected field name") ||
        moduleContext.resolveSymbolEntry(result, symtabEntry, fieldName, loc))
      return failure();
    break;
  }
  }
  // Don't add code here, the common cases of these switch statements will be
  // merged. This allows for fixing up primops after they have been created.
  switch (kind) {
  case FIRToken::lp_shr:
    // For FIRRTL versions earlier than 4.0.0, insert pad(_, 1) around any
    // unsigned shr This ensures the minimum width is 1 (but can be greater)
    if (version < FIRVersion(4, 0, 0) && type_isa<UIntType>(result.getType()))
      result = PadPrimOp::create(builder, result, 1);
    break;
  default:
    break;
  }

  return parseOptionalExpPostscript(result);
}

/// Parse the postfix productions of expression after the leading expression
/// has been parsed.
///
///  exp ::= exp '.' fieldId
///      ::= exp '[' intLit ']'
/// XX   ::= exp '.' DoubleLit // TODO Workaround for #470
///      ::= exp '[' exp ']'
ParseResult FIRStmtParser::parseOptionalExpPostscript(Value &result,
                                                      bool allowDynamic) {

  // Handle postfix expressions.
  while (true) {
    // Subfield: exp ::= exp '.' fieldId
    if (consumeIf(FIRToken::period)) {
      if (parsePostFixFieldId(result))
        return failure();

      continue;
    }

    // Subindex: exp ::= exp '[' intLit ']' | exp '[' exp ']'
    if (consumeIf(FIRToken::l_square)) {
      if (getToken().isAny(FIRToken::integer, FIRToken::string)) {
        if (parsePostFixIntSubscript(result))
          return failure();
        continue;
      }
      if (!allowDynamic)
        return emitError("subaccess not allowed here");
      if (parsePostFixDynamicSubscript(result))
        return failure();

      continue;
    }

    return success();
  }
}

template <typename subop>
FailureOr<Value>
FIRStmtParser::emitCachedSubAccess(Value base, unsigned indexNo, SMLoc loc) {
  // Check if we already have created this Subindex op.
  auto &value = moduleContext.getCachedSubaccess(base, indexNo);
  if (value)
    return value;

  // Make sure the field name matches up with the input value's type and
  // compute the result type for the expression.
  auto baseType = cast<FIRRTLType>(base.getType());
  auto resultType = subop::inferReturnType(baseType, indexNo, {});
  if (!resultType) {
    // Emit the error at the right location.  translateLocation is expensive.
    (void)subop::inferReturnType(baseType, indexNo, translateLocation(loc));
    return failure();
  }

  // Create the result operation, inserting at the location of the declaration.
  // This will cache the subfield operation for further uses.
  locationProcessor.setLoc(loc);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(base);
  auto op = subop::create(builder, resultType, base, indexNo);

  // Insert the newly created operation into the cache.
  return value = op.getResult();
}

/// exp ::= exp '.' fieldId
///
/// The "exp '.'" part of the production has already been parsed.
///
ParseResult FIRStmtParser::parsePostFixFieldId(Value &result) {
  auto loc = getToken().getLoc();
  SmallVector<StringRef, 3> fields;
  if (parseFieldIdSeq(fields, "expected field name"))
    return failure();
  for (auto fieldName : fields) {
    std::optional<unsigned> indexV;
    auto type = result.getType();
    if (auto refTy = type_dyn_cast<RefType>(type))
      type = refTy.getType();
    if (auto bundle = type_dyn_cast<BundleType>(type))
      indexV = bundle.getElementIndex(fieldName);
    else if (auto bundle = type_dyn_cast<OpenBundleType>(type))
      indexV = bundle.getElementIndex(fieldName);
    else if (auto klass = type_dyn_cast<ClassType>(type))
      indexV = klass.getElementIndex(fieldName);
    else
      return emitError(loc, "subfield requires bundle or object operand ");
    if (!indexV)
      return emitError(loc, "unknown field '" + fieldName + "' in type ")
             << result.getType();
    auto indexNo = *indexV;

    FailureOr<Value> subResult;
    if (type_isa<RefType>(result.getType()))
      subResult = emitCachedSubAccess<RefSubOp>(result, indexNo, loc);
    else if (type_isa<ClassType>(type))
      subResult = emitCachedSubAccess<ObjectSubfieldOp>(result, indexNo, loc);
    else if (type_isa<BundleType>(type))
      subResult = emitCachedSubAccess<SubfieldOp>(result, indexNo, loc);
    else
      subResult = emitCachedSubAccess<OpenSubfieldOp>(result, indexNo, loc);

    if (failed(subResult))
      return failure();
    result = *subResult;
  }
  return success();
}

/// exp ::= exp '[' intLit ']'
///
/// The "exp '['" part of the production has already been parsed.
///
ParseResult FIRStmtParser::parsePostFixIntSubscript(Value &result) {
  auto loc = getToken().getLoc();
  int32_t indexNo;
  if (parseIntLit(indexNo, "expected index") ||
      parseToken(FIRToken::r_square, "expected ']'"))
    return failure();

  if (indexNo < 0)
    return emitError(loc, "invalid index specifier"), failure();

  FailureOr<Value> subResult;
  if (type_isa<RefType>(result.getType()))
    subResult = emitCachedSubAccess<RefSubOp>(result, indexNo, loc);
  else if (type_isa<FVectorType>(result.getType()))
    subResult = emitCachedSubAccess<SubindexOp>(result, indexNo, loc);
  else
    subResult = emitCachedSubAccess<OpenSubindexOp>(result, indexNo, loc);

  if (failed(subResult))
    return failure();
  result = *subResult;
  return success();
}

/// exp ::= exp '[' exp ']'
///
/// The "exp '['" part of the production has already been parsed.
///
ParseResult FIRStmtParser::parsePostFixDynamicSubscript(Value &result) {
  auto loc = getToken().getLoc();
  Value index;
  if (parseExp(index, "expected subscript index expression") ||
      parseToken(FIRToken::r_square, "expected ']' in subscript"))
    return failure();

  // If the index expression is a flip type, strip it off.
  auto indexType = type_dyn_cast<FIRRTLBaseType>(index.getType());
  if (!indexType)
    return emitError("expected base type for index expression");
  indexType = indexType.getPassiveType();
  locationProcessor.setLoc(loc);

  // Make sure the index expression is valid and compute the result type for the
  // expression.
  auto resultType =
      SubaccessOp::inferReturnType(result.getType(), index.getType(), {});
  if (!resultType) {
    // Emit the error at the right location.  translateLocation is expensive.
    (void)SubaccessOp::inferReturnType(result.getType(), index.getType(),
                                       translateLocation(loc));
    return failure();
  }

  // Create the result operation.
  auto op = SubaccessOp::create(builder, resultType, result, index);
  result = op.getResult();
  return success();
}

/// integer-literal-exp ::= 'UInt' optional-width '(' intLit ')'
///                     ::= 'SInt' optional-width '(' intLit ')'
ParseResult FIRStmtParser::parseIntegerLiteralExp(Value &result) {
  bool isSigned = getToken().is(FIRToken::kw_SInt);
  auto loc = getToken().getLoc();
  consumeToken();

  // Parse a width specifier if present.
  int32_t width;
  APInt value;
  if (parseOptionalWidth(width) ||
      parseToken(FIRToken::l_paren, "expected '(' in integer expression") ||
      parseIntLit(value, "expected integer value") ||
      parseToken(FIRToken::r_paren, "expected ')' in integer expression"))
    return failure();

  // Construct an integer attribute of the right width.
  // Literals are parsed as 'const' types.
  auto type = IntType::get(builder.getContext(), isSigned, width, true);

  IntegerType::SignednessSemantics signedness =
      isSigned ? IntegerType::Signed : IntegerType::Unsigned;
  if (width == 0) {
    if (!value.isZero())
      return emitError(loc, "zero bit constant must be zero");
    value = value.trunc(0);
  } else if (width != -1) {
    // Convert to the type's width, checking value fits in destination width.
    bool valueFits = isSigned ? value.isSignedIntN(width) : value.isIntN(width);
    if (!valueFits)
      return emitError(loc, "initializer too wide for declared width");
    value = isSigned ? value.sextOrTrunc(width) : value.zextOrTrunc(width);
  }

  Type attrType =
      IntegerType::get(type.getContext(), value.getBitWidth(), signedness);
  auto attr = builder.getIntegerAttr(attrType, value);

  locationProcessor.setLoc(loc);
  result = moduleContext.getCachedConstant(builder, attr, type, attr);
  return success();
}

/// list-exp ::= list-type '(' exp* ')'
ParseResult FIRStmtParser::parseListExp(Value &result) {
  auto loc = getToken().getLoc();
  FIRRTLType type;
  if (parseListType(type))
    return failure();
  auto listType = type_cast<ListType>(type);
  auto elementType = listType.getElementType();

  if (parseToken(FIRToken::l_paren, "expected '(' in List expression"))
    return failure();

  SmallVector<Value, 3> operands;
  if (parseListUntil(FIRToken::r_paren, [&]() -> ParseResult {
        Value operand;
        locationProcessor.setLoc(loc);
        if (parseExp(operand, "expected expression in List expression"))
          return failure();

        if (operand.getType() != elementType) {
          if (!isa<AnyRefType>(elementType) ||
              !isa<ClassType>(operand.getType()))
            return emitError(loc, "unexpected expression of type ")
                   << operand.getType() << " in List expression of type "
                   << elementType;
          operand = ObjectAnyRefCastOp::create(builder, operand);
        }

        operands.push_back(operand);
        return success();
      }))
    return failure();

  locationProcessor.setLoc(loc);
  result = ListCreateOp::create(builder, listType, operands);
  return success();
}

/// list-concat-exp ::= 'list_concat' '(' exp* ')'
ParseResult FIRStmtParser::parseListConcatExp(Value &result) {
  consumeToken(FIRToken::lp_list_concat);

  auto loc = getToken().getLoc();
  ListType type;
  SmallVector<Value, 3> operands;
  if (parseListUntil(FIRToken::r_paren, [&]() -> ParseResult {
        Value operand;
        locationProcessor.setLoc(loc);
        if (parseExp(operand, "expected expression in List concat expression"))
          return failure();

        if (!type_isa<ListType>(operand.getType()))
          return emitError(loc, "unexpected expression of type ")
                 << operand.getType() << " in List concat expression";

        if (!type)
          type = type_cast<ListType>(operand.getType());

        if (operand.getType() != type)
          return emitError(loc, "unexpected expression of type ")
                 << operand.getType() << " in List concat expression of type "
                 << type;

        operands.push_back(operand);
        return success();
      }))
    return failure();

  if (operands.empty())
    return emitError(loc, "need at least one List to concatenate");

  locationProcessor.setLoc(loc);
  result = ListConcatOp::create(builder, type, operands);
  return success();
}

/// cat-exp ::= 'cat(' exp* ')'
ParseResult FIRStmtParser::parseCatExp(Value &result) {
  consumeToken(FIRToken::lp_cat);

  auto loc = getToken().getLoc();
  SmallVector<Value, 3> operands;
  std::optional<bool> isSigned;
  if (parseListUntil(FIRToken::r_paren, [&]() -> ParseResult {
        Value operand;
        locationProcessor.setLoc(loc);
        auto operandLoc = getToken().getLoc();
        if (parseExp(operand, "expected expression in cat expression"))
          return failure();
        if (!type_isa<IntType>(operand.getType())) {
          auto diag = emitError(loc, "all operands must be Int type");
          diag.attachNote(translateLocation(operandLoc))
              << "non-integer operand is here";
          return failure();
        }
        if (!isSigned)
          isSigned = type_isa<SIntType>(operand.getType());
        else if (type_isa<SIntType>(operand.getType()) != *isSigned) {
          auto diag = emitError(loc, "all operands must have same signedness");
          diag.attachNote(translateLocation(operandLoc))
              << "operand with different signedness is here";
          return failure();
        }

        operands.push_back(operand);
        return success();
      }))
    return failure();

  if (operands.size() != 2) {
    if (requireFeature(nextFIRVersion, "variadic cat", loc))
      return failure();
  }

  locationProcessor.setLoc(loc);
  result = CatPrimOp::create(builder, operands);
  return success();
}

/// The .fir grammar has the annoying property where:
/// 1) some statements start with keywords
/// 2) some start with an expression
/// 3) it allows the 'reference' expression to either be an identifier or a
///    keyword.
///
/// One example of this is something like, where this is not a register decl:
///   reg <- thing
///
/// Solving this requires lookahead to the second token.  We handle it by
///  factoring the lookahead inline into the code to keep the parser fast.
///
/// As such, statements that start with a leading keyword call this method to
/// check to see if the keyword they consumed was actually the start of an
/// expression.  If so, they parse the expression-based statement and return the
/// parser result.  If not, they return None and the statement is parsed like
/// normal.
std::optional<ParseResult>
FIRStmtParser::parseExpWithLeadingKeyword(FIRToken keyword) {
  switch (getToken().getKind()) {
  default:
    // This isn't part of an expression, and isn't part of a statement.
    return std::nullopt;

  case FIRToken::period:     // exp `.` identifier
  case FIRToken::l_square:   // exp `[` index `]`
  case FIRToken::kw_is:      // exp is invalid
  case FIRToken::less_equal: // exp <= thing
    break;
  }

  Value lhs;
  SymbolValueEntry symtabEntry;
  auto loc = keyword.getLoc();

  if (moduleContext.lookupSymbolEntry(symtabEntry, keyword.getSpelling(), loc))
    return ParseResult(failure());

  // If we have a '.', we might have a symbol or an expanded port.  If we
  // resolve to a symbol, use that, otherwise check for expanded bundles of
  // other ops.
  // Non '.' ops take the plain symbol path.
  if (moduleContext.resolveSymbolEntry(lhs, symtabEntry, loc, false)) {
    // Ok if the base name didn't resolve by itself, it might be part of an
    // expanded dot reference.  That doesn't work then we fail.
    if (!consumeIf(FIRToken::period))
      return ParseResult(failure());

    StringRef fieldName;
    if (parseFieldId(fieldName, "expected field name") ||
        moduleContext.resolveSymbolEntry(lhs, symtabEntry, fieldName, loc))
      return ParseResult(failure());
  }

  // Parse any further trailing things like "mem.x.y".
  if (parseOptionalExpPostscript(lhs))
    return ParseResult(failure());

  return parseLeadingExpStmt(lhs);
}
//===-----------------------------
// FIRStmtParser Statement Parsing

/// simple_stmt_block ::= simple_stmt*
ParseResult FIRStmtParser::parseSimpleStmtBlock(unsigned indent) {
  while (true) {
    // The outer level parser can handle these tokens.
    if (getToken().isAny(FIRToken::eof, FIRToken::error))
      return success();

    auto subIndent = getIndentation();
    if (!subIndent.has_value())
      return emitError("expected statement to be on its own line"), failure();

    if (*subIndent <= indent)
      return success();

    // Let the statement parser handle this.
    if (parseSimpleStmt(*subIndent))
      return failure();
  }
}

ParseResult FIRStmtParser::parseSimpleStmt(unsigned stmtIndent) {
  locationProcessor.startStatement();
  auto result = parseSimpleStmtImpl(stmtIndent);
  locationProcessor.endStatement(*this);
  return result;
}

/// simple_stmt ::= stmt
///
/// stmt ::= attach
///      ::= memport
///      ::= printf
///      ::= skip
///      ::= stop
///      ::= when
///      ::= leading-exp-stmt
///      ::= define
///      ::= propassign
///
/// stmt ::= instance
///      ::= cmem | smem | mem
///      ::= node | wire
///      ::= register
///      ::= contract
///
ParseResult FIRStmtParser::parseSimpleStmtImpl(unsigned stmtIndent) {
  auto kind = getToken().getKind();
  /// Massage the kind based on the FIRRTL Version.
  switch (kind) {
  case FIRToken::kw_invalidate:
  case FIRToken::kw_connect:
  case FIRToken::kw_regreset:
    /// The "invalidate", "connect", and "regreset" keywords were added
    /// in 3.0.0.
    if (version < FIRVersion(3, 0, 0))
      kind = FIRToken::identifier;
    break;
  default:
    break;
  };
  switch (kind) {
  // Statements.
  case FIRToken::kw_attach:
    return parseAttach();
  case FIRToken::kw_infer:
    return parseMemPort(MemDirAttr::Infer);
  case FIRToken::kw_read:
    return parseMemPort(MemDirAttr::Read);
  case FIRToken::kw_write:
    return parseMemPort(MemDirAttr::Write);
  case FIRToken::kw_rdwr:
    return parseMemPort(MemDirAttr::ReadWrite);
  case FIRToken::kw_connect:
    return parseConnect();
  case FIRToken::kw_propassign:
    if (requireFeature({3, 1, 0}, "properties"))
      return failure();
    return parsePropAssign();
  case FIRToken::kw_invalidate:
    return parseInvalidate();
  case FIRToken::lp_printf:
    return parsePrintf();
  case FIRToken::lp_fprintf:
    return parseFPrintf();
  case FIRToken::lp_fflush:
    return parseFFlush();
  case FIRToken::kw_skip:
    return parseSkip();
  case FIRToken::lp_stop:
    return parseStop();
  case FIRToken::lp_assert:
    return parseAssert();
  case FIRToken::lp_assume:
    return parseAssume();
  case FIRToken::lp_cover:
    return parseCover();
  case FIRToken::kw_when:
    return parseWhen(stmtIndent);
  case FIRToken::kw_match:
    return parseMatch(stmtIndent);
  case FIRToken::kw_define:
    return parseRefDefine();
  case FIRToken::lp_force:
    return parseRefForce();
  case FIRToken::lp_force_initial:
    return parseRefForceInitial();
  case FIRToken::lp_release:
    return parseRefRelease();
  case FIRToken::lp_release_initial:
    return parseRefReleaseInitial();
  case FIRToken::kw_group:
    if (requireFeature({3, 2, 0}, "optional groups") ||
        removedFeature({3, 3, 0}, "optional groups"))
      return failure();
    return parseLayerBlockOrGroup(stmtIndent);
  case FIRToken::kw_layerblock:
    if (requireFeature({3, 3, 0}, "layers"))
      return failure();
    return parseLayerBlockOrGroup(stmtIndent);
  case FIRToken::lp_intrinsic:
    if (requireFeature({4, 0, 0}, "generic intrinsics"))
      return failure();
    return parseIntrinsicStmt();
  default: {
    // Statement productions that start with an expression.
    Value lhs;
    if (parseExpLeadingStmt(lhs, "unexpected token in module"))
      return failure();
    // We use parseExp in a special mode that can complete the entire stmt
    // at once in unusual cases.  If this happened, then we are done.
    if (!lhs)
      return success();

    return parseLeadingExpStmt(lhs);
  }

    // Declarations
  case FIRToken::kw_inst:
    return parseInstance();
  case FIRToken::kw_instchoice:
    return parseInstanceChoice();
  case FIRToken::kw_object:
    return parseObject();
  case FIRToken::kw_cmem:
    return parseCombMem();
  case FIRToken::kw_smem:
    return parseSeqMem();
  case FIRToken::kw_mem:
    return parseMem(stmtIndent);
  case FIRToken::kw_node:
    return parseNode();
  case FIRToken::kw_wire:
    return parseWire();
  case FIRToken::kw_reg:
    return parseRegister(stmtIndent);
  case FIRToken::kw_regreset:
    return parseRegisterWithReset();
  case FIRToken::kw_contract:
    return parseContract(stmtIndent);
  }
}

ParseResult FIRStmtParser::parseSubBlock(Block &blockToInsertInto,
                                         unsigned indent,
                                         SymbolRefAttr layerSym) {
  // Declarations within the suite are scoped to within the suite.
  auto suiteScope = std::make_unique<FIRModuleContext::ContextScope>(
      moduleContext, &blockToInsertInto);

  // After parsing the when region, we can release any new entries in
  // unbundledValues since the symbol table entries that refer to them will be
  // gone.
  UnbundledValueRestorer x(moduleContext.unbundledValues);

  // We parse the substatements into their own parser, so they get inserted
  // into the specified 'when' region.
  auto subParser = std::make_unique<FIRStmtParser>(
      blockToInsertInto, moduleContext, innerSymFixups, circuitSymTbl, version,
      layerSym);

  // Figure out whether the body is a single statement or a nested one.
  auto stmtIndent = getIndentation();

  // Parsing a single statment is straightforward.
  if (!stmtIndent.has_value())
    return subParser->parseSimpleStmt(indent);

  if (*stmtIndent <= indent)
    return emitError("statement must be indented more than previous statement"),
           failure();

  // Parse a block of statements that are indented more than the when.
  return subParser->parseSimpleStmtBlock(indent);
}

/// attach ::= 'attach' '(' exp+ ')' info?
ParseResult FIRStmtParser::parseAttach() {
  auto startTok = consumeToken(FIRToken::kw_attach);

  // If this was actually the start of a connect or something else handle that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  if (parseToken(FIRToken::l_paren, "expected '(' after attach"))
    return failure();

  SmallVector<Value, 4> operands;
  operands.push_back({});
  if (parseExp(operands.back(), "expected operand in attach"))
    return failure();

  while (consumeIf(FIRToken::comma)) {
    operands.push_back({});
    if (parseExp(operands.back(), "expected operand in attach"))
      return failure();
  }
  if (parseToken(FIRToken::r_paren, "expected close paren"))
    return failure();

  if (parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());
  AttachOp::create(builder, operands);
  return success();
}

/// stmt ::= mdir 'mport' id '=' id '[' exp ']' exp info?
/// mdir ::= 'infer' | 'read' | 'write' | 'rdwr'
///
ParseResult FIRStmtParser::parseMemPort(MemDirAttr direction) {
  auto startTok = consumeToken();
  auto startLoc = startTok.getLoc();

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  StringRef id;
  StringRef memName;
  SymbolValueEntry memorySym;
  Value memory, indexExp, clock;
  if (parseToken(FIRToken::kw_mport, "expected 'mport' in memory port") ||
      parseId(id, "expected result name") ||
      parseToken(FIRToken::equal, "expected '=' in memory port") ||
      parseId(memName, "expected memory name") ||
      moduleContext.lookupSymbolEntry(memorySym, memName, startLoc) ||
      moduleContext.resolveSymbolEntry(memory, memorySym, startLoc) ||
      parseToken(FIRToken::l_square, "expected '[' in memory port") ||
      parseExp(indexExp, "expected index expression") ||
      parseToken(FIRToken::r_square, "expected ']' in memory port") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(clock, "expected clock expression") || parseOptionalInfo())
    return failure();

  auto memVType = type_dyn_cast<CMemoryType>(memory.getType());
  if (!memVType)
    return emitError(startLoc,
                     "memory port should have behavioral memory type");
  auto resultType = memVType.getElementType();

  ArrayAttr annotations = getConstants().emptyArrayAttr;
  locationProcessor.setLoc(startLoc);

  // Create the memory port at the location of the cmemory.
  Value memoryPort, memoryData;
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfterValue(memory);
    auto memoryPortOp = MemoryPortOp::create(
        builder, resultType, CMemoryPortType::get(getContext()), memory,
        direction, id, annotations);
    memoryData = memoryPortOp.getResult(0);
    memoryPort = memoryPortOp.getResult(1);
  }

  // Create a memory port access in the current scope.
  MemoryPortAccessOp::create(builder, memoryPort, indexExp, clock);

  return moduleContext.addSymbolEntry(id, memoryData, startLoc, true);
}

// Parse a format string and build operations for FIRRTL "special"
// substitutions. Set `formatStringResult` to the validated format string and
// `operands` to the list of actual operands.
ParseResult FIRStmtParser::parseFormatString(SMLoc formatStringLoc,
                                             StringRef formatString,
                                             ArrayRef<Value> specOperands,
                                             StringAttr &formatStringResult,
                                             SmallVectorImpl<Value> &operands) {
  // Validate the format string and remove any "special" substitutions.  Only do
  // this for FIRRTL versions > 5.0.0.  If at a different FIRRTL version, then
  // just parse this as if it was a string.
  SmallVector<Attribute, 4> specialSubstitutions;
  SmallString<64> validatedFormatString;
  if (version < FIRVersion(5, 0, 0)) {
    validatedFormatString = formatString;
    operands.append(specOperands.begin(), specOperands.end());
  } else {
    for (size_t i = 0, e = formatString.size(), opIdx = 0; i != e; ++i) {
      auto c = formatString[i];
      switch (c) {
      // FIRRTL percent format strings.  If this is actually a format string,
      // then grab one of the "spec" operands.
      case '%': {
        validatedFormatString.push_back(c);

        // Parse the width specifier.
        SmallString<6> width;
        c = formatString[++i];
        while (isdigit(c)) {
          width.push_back(c);
          c = formatString[++i];
        }

        // Parse the radix.
        switch (c) {
        case 'c':
          if (!width.empty()) {
            emitError(formatStringLoc) << "ASCII character format specifiers "
                                          "('%c') may not specify a width";
            return failure();
          }
          [[fallthrough]];
        case 'b':
        case 'd':
        case 'x':
          if (!width.empty())
            validatedFormatString.append(width);
          operands.push_back(specOperands[opIdx++]);
          break;
        case '%':
          if (!width.empty()) {
            emitError(formatStringLoc)
                << "literal percents ('%%') may not specify a width";
            return failure();
          }
          break;
        // Anything else is illegal.
        default:
          emitError(formatStringLoc)
              << "unknown printf substitution '%" << width << c << "'";
          return failure();
        }
        validatedFormatString.push_back(c);
        break;
      }
      // FIRRTL special format strings.  If this is a special format string,
      // then create an operation for it and put its result in the operand list.
      // This will cause the operands to interleave with the spec operands.
      // Replace any special format string with the generic '{{}}' placeholder.
      case '{': {
        if (formatString[i + 1] != '{') {
          validatedFormatString.push_back(c);
          break;
        }
        // Handle a special substitution.
        i += 2;
        size_t start = i;
        while (formatString[i] != '}')
          ++i;
        if (formatString[i] != '}') {
          llvm::errs() << "expected '}' to terminate special substitution";
          return failure();
        }

        auto specialString = formatString.slice(start, i);
        if (specialString == "SimulationTime") {
          operands.push_back(TimeOp::create(builder));
        } else if (specialString == "HierarchicalModuleName") {
          operands.push_back(HierarchicalModuleNameOp::create(builder));
        } else {
          emitError(formatStringLoc)
              << "unknown printf substitution '" << specialString
              << "' (did you misspell it?)";
          return failure();
        }

        validatedFormatString.append("{{}}");
        ++i;
        break;
      }
      default:
        validatedFormatString.push_back(c);
      }
    }
  }

  formatStringResult =
      builder.getStringAttr(FIRToken::getStringValue(validatedFormatString));
  return success();
}

/// printf ::= 'printf(' exp exp StringLit exp* ')' name? info?
ParseResult FIRStmtParser::parsePrintf() {
  auto startTok = consumeToken(FIRToken::lp_printf);

  Value clock, condition;
  StringRef formatString;
  if (parseExp(clock, "expected clock expression in printf") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(condition, "expected condition in printf") ||
      parseToken(FIRToken::comma, "expected ','"))
    return failure();

  auto formatStringLoc = getToken().getLoc();
  if (parseGetSpelling(formatString) ||
      parseToken(FIRToken::string, "expected format string in printf"))
    return failure();

  SmallVector<Value, 4> specOperands;
  while (consumeIf(FIRToken::comma)) {
    specOperands.push_back({});
    if (parseExp(specOperands.back(), "expected operand in printf"))
      return failure();
  }

  StringAttr name;
  if (parseToken(FIRToken::r_paren, "expected ')'") ||
      parseOptionalName(name) || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  StringAttr formatStrUnescaped;
  SmallVector<Value> operands;
  if (parseFormatString(formatStringLoc, formatString, specOperands,
                        formatStrUnescaped, operands))
    return failure();

  PrintFOp::create(builder, clock, condition, formatStrUnescaped, operands,
                   name);
  return success();
}

/// fprintf ::= 'fprintf(' exp exp StringLit StringLit exp* ')' name? info?
ParseResult FIRStmtParser::parseFPrintf() {
  if (requireFeature(nextFIRVersion, "fprintf"))
    return failure();
  auto startTok = consumeToken(FIRToken::lp_fprintf);

  Value clock, condition;
  StringRef outputFile, formatString;
  if (parseExp(clock, "expected clock expression in fprintf") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(condition, "expected condition in fprintf") ||
      parseToken(FIRToken::comma, "expected ','"))
    return failure();

  auto outputFileLoc = getToken().getLoc();
  if (parseGetSpelling(outputFile) ||
      parseToken(FIRToken::string, "expected output file in fprintf"))
    return failure();

  SmallVector<Value, 4> outputFileSpecOperands;
  while (consumeIf(FIRToken::comma)) {
    // Stop parsing operands when we see the format string.
    if (getToken().getKind() == FIRToken::string)
      break;
    outputFileSpecOperands.push_back({});
    if (parseExp(outputFileSpecOperands.back(), "expected operand in fprintf"))
      return failure();
  }

  auto formatStringLoc = getToken().getLoc();
  if (parseGetSpelling(formatString) ||
      parseToken(FIRToken::string, "expected format string in printf"))
    return failure();

  SmallVector<Value, 4> specOperands;
  while (consumeIf(FIRToken::comma)) {
    specOperands.push_back({});
    if (parseExp(specOperands.back(), "expected operand in fprintf"))
      return failure();
  }

  StringAttr name;
  if (parseToken(FIRToken::r_paren, "expected ')'") ||
      parseOptionalName(name) || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  StringAttr outputFileNameStrUnescaped;
  SmallVector<Value> outputFileOperands;
  if (parseFormatString(outputFileLoc, outputFile, outputFileSpecOperands,
                        outputFileNameStrUnescaped, outputFileOperands))
    return failure();

  StringAttr formatStrUnescaped;
  SmallVector<Value> operands;
  if (parseFormatString(formatStringLoc, formatString, specOperands,
                        formatStrUnescaped, operands))
    return failure();

  FPrintFOp::create(builder, clock, condition, outputFileNameStrUnescaped,
                    outputFileOperands, formatStrUnescaped, operands, name);
  return success();
}

/// fflush ::= 'fflush(' exp exp (StringLit exp*)? ')' info?
ParseResult FIRStmtParser::parseFFlush() {
  if (requireFeature(nextFIRVersion, "fflush"))
    return failure();

  auto startTok = consumeToken(FIRToken::lp_fflush);

  Value clock, condition;
  if (parseExp(clock, "expected clock expression in 'fflush'") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(condition, "expected condition in 'fflush'"))
    return failure();

  locationProcessor.setLoc(startTok.getLoc());
  StringAttr outputFileNameStrUnescaped;
  SmallVector<Value> outputFileOperands;
  // Parse file name if present.
  if (consumeIf(FIRToken::comma)) {
    SmallVector<Value, 4> outputFileSpecOperands;
    auto outputFileLoc = getToken().getLoc();
    StringRef outputFile;
    if (parseGetSpelling(outputFile) ||
        parseToken(FIRToken::string, "expected output file in fflush"))
      return failure();

    while (consumeIf(FIRToken::comma)) {
      outputFileSpecOperands.push_back({});
      if (parseExp(outputFileSpecOperands.back(), "expected operand in fflush"))
        return failure();
    }

    if (parseFormatString(outputFileLoc, outputFile, outputFileSpecOperands,
                          outputFileNameStrUnescaped, outputFileOperands))
      return failure();
  }

  if (parseToken(FIRToken::r_paren, "expected ')' in 'fflush'") ||
      parseOptionalInfo())
    return failure();

  FFlushOp::create(builder, clock, condition, outputFileNameStrUnescaped,
                   outputFileOperands);
  return success();
}

/// skip ::= 'skip' info?
ParseResult FIRStmtParser::parseSkip() {
  auto startTok = consumeToken(FIRToken::kw_skip);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  if (parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());
  SkipOp::create(builder);
  return success();
}

/// stop ::= 'stop(' exp exp intLit ')' info?
ParseResult FIRStmtParser::parseStop() {
  auto startTok = consumeToken(FIRToken::lp_stop);

  Value clock, condition;
  int64_t exitCode;
  StringAttr name;
  if (parseExp(clock, "expected clock expression in 'stop'") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(condition, "expected condition in 'stop'") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseIntLit(exitCode, "expected exit code in 'stop'") ||
      parseToken(FIRToken::r_paren, "expected ')' in 'stop'") ||
      parseOptionalName(name) || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());
  StopOp::create(builder, clock, condition, builder.getI32IntegerAttr(exitCode),
                 name);
  return success();
}

/// assert ::= 'assert(' exp exp exp StringLit exp*')' info?
ParseResult FIRStmtParser::parseAssert() {
  auto startTok = consumeToken(FIRToken::lp_assert);

  Value clock, predicate, enable;
  StringRef formatString;
  StringAttr name;
  if (parseExp(clock, "expected clock expression in 'assert'") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(predicate, "expected predicate in 'assert'") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(enable, "expected enable in 'assert'") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseGetSpelling(formatString) ||
      parseToken(FIRToken::string, "expected format string in 'assert'"))
    return failure();

  SmallVector<Value, 4> operands;
  while (!consumeIf(FIRToken::r_paren)) {
    operands.push_back({});
    if (parseToken(FIRToken::comma, "expected ','") ||
        parseExp(operands.back(), "expected operand in 'assert'"))
      return failure();
  }

  if (parseOptionalName(name) || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());
  auto formatStrUnescaped = FIRToken::getStringValue(formatString);
  AssertOp::create(builder, clock, predicate, enable, formatStrUnescaped,
                   operands, name.getValue());
  return success();
}

/// assume ::= 'assume(' exp exp exp StringLit exp* ')' info?
ParseResult FIRStmtParser::parseAssume() {
  auto startTok = consumeToken(FIRToken::lp_assume);

  Value clock, predicate, enable;
  StringRef formatString;
  StringAttr name;
  if (parseExp(clock, "expected clock expression in 'assume'") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(predicate, "expected predicate in 'assume'") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(enable, "expected enable in 'assume'") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseGetSpelling(formatString) ||
      parseToken(FIRToken::string, "expected format string in 'assume'"))
    return failure();

  SmallVector<Value, 4> operands;
  while (!consumeIf(FIRToken::r_paren)) {
    operands.push_back({});
    if (parseToken(FIRToken::comma, "expected ','") ||
        parseExp(operands.back(), "expected operand in 'assume'"))
      return failure();
  }

  if (parseOptionalName(name) || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());
  auto formatStrUnescaped = FIRToken::getStringValue(formatString);
  AssumeOp::create(builder, clock, predicate, enable, formatStrUnescaped,
                   operands, name.getValue());
  return success();
}

/// cover ::= 'cover(' exp exp exp StringLit ')' info?
ParseResult FIRStmtParser::parseCover() {
  auto startTok = consumeToken(FIRToken::lp_cover);

  Value clock, predicate, enable;
  StringRef message;
  StringAttr name;
  if (parseExp(clock, "expected clock expression in 'cover'") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(predicate, "expected predicate in 'cover'") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(enable, "expected enable in 'cover'") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseGetSpelling(message) ||
      parseToken(FIRToken::string, "expected message in 'cover'") ||
      parseToken(FIRToken::r_paren, "expected ')' in 'cover'") ||
      parseOptionalName(name) || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());
  auto messageUnescaped = FIRToken::getStringValue(message);
  CoverOp::create(builder, clock, predicate, enable, messageUnescaped,
                  ValueRange{}, name.getValue());
  return success();
}

/// when  ::= 'when' exp ':' info? suite? ('else' ( when | ':' info? suite?)
/// )? suite ::= simple_stmt | INDENT simple_stmt+ DEDENT
ParseResult FIRStmtParser::parseWhen(unsigned whenIndent) {
  auto startTok = consumeToken(FIRToken::kw_when);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  Value condition;
  if (parseExp(condition, "expected condition in 'when'") ||
      parseToken(FIRToken::colon, "expected ':' in when") ||
      parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());
  // Create the IR representation for the when.
  auto whenStmt = WhenOp::create(builder, condition, /*createElse*/ false);

  // Parse the 'then' body into the 'then' region.
  if (parseSubBlock(whenStmt.getThenBlock(), whenIndent, layerSym))
    return failure();

  // If the else is present, handle it otherwise we're done.
  if (getToken().isNot(FIRToken::kw_else))
    return success();

  // If the 'else' is less indented than the when, then it must belong to some
  // containing 'when'.
  auto elseIndent = getIndentation();
  if (elseIndent && *elseIndent < whenIndent)
    return success();

  consumeToken(FIRToken::kw_else);

  // Create an else block to parse into.
  whenStmt.createElseRegion();

  // If we have the ':' form, then handle it.

  // Syntactic shorthand 'else when'. This uses the same indentation level as
  // the outer 'when'.
  if (getToken().is(FIRToken::kw_when)) {
    // We create a sub parser for the else block.
    auto subParser = std::make_unique<FIRStmtParser>(
        whenStmt.getElseBlock(), moduleContext, innerSymFixups, circuitSymTbl,
        version, layerSym);

    return subParser->parseSimpleStmt(whenIndent);
  }

  // Parse the 'else' body into the 'else' region.
  LocationAttr elseLoc; // ignore the else locator.
  if (parseToken(FIRToken::colon, "expected ':' after 'else'") ||
      parseOptionalInfoLocator(elseLoc) ||
      parseSubBlock(whenStmt.getElseBlock(), whenIndent, layerSym))
    return failure();

  // TODO(firrtl spec): There is no reason for the 'else :' grammar to take an
  // info.  It doesn't appear to be generated either.
  return success();
}

/// enum-exp ::= enum-type '(' Id ( ',' exp )? ')'
ParseResult FIRStmtParser::parseEnumExp(Value &value) {
  auto startLoc = getToken().getLoc();
  locationProcessor.setLoc(startLoc);
  FIRRTLType type;
  if (parseEnumType(type))
    return failure();

  // Check that the input type is a legal enumeration.
  auto enumType = type_dyn_cast<FEnumType>(type);
  if (!enumType)
    return emitError(startLoc,
                     "expected enumeration type in enumeration expression");

  StringRef tag;
  if (parseToken(FIRToken::l_paren, "expected '(' in enumeration expression") ||
      parseId(tag, "expected enumeration tag"))
    return failure();

  Value input;
  if (consumeIf(FIRToken::r_paren)) {
    // If the payload is not specified, we create a 0 bit unsigned integer
    // constant.
    auto type = IntType::get(builder.getContext(), false, 0, true);
    Type attrType = IntegerType::get(getContext(), 0, IntegerType::Unsigned);
    auto attr = builder.getIntegerAttr(attrType, APInt(0, 0, false));
    input = ConstantOp::create(builder, type, attr);
  } else {
    // Otherwise we parse an expression.
    if (parseToken(FIRToken::comma, "expected ','") ||
        parseExp(input, "expected expression in enumeration value") ||
        parseToken(FIRToken::r_paren, "expected closing ')'"))
      return failure();
  }

  value = FEnumCreateOp::create(builder, enumType, tag, input);
  return success();
}

/// match ::= 'match' exp ':' info?
///             (INDENT ( Id ( '(' Id ')' )? ':'
///               (INDENT simple_stmt* DEDENT )?
///             )* DEDENT)?
ParseResult FIRStmtParser::parseMatch(unsigned matchIndent) {
  auto startTok = consumeToken(FIRToken::kw_match);

  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  Value input;
  if (parseExp(input, "expected expression in 'match'") ||
      parseToken(FIRToken::colon, "expected ':' in 'match'") ||
      parseOptionalInfo())
    return failure();

  auto enumType = type_dyn_cast<FEnumType>(input.getType());
  if (!enumType)
    return mlir::emitError(
               input.getLoc(),
               "expected enumeration type for 'match' statement, but got ")
           << input.getType();

  locationProcessor.setLoc(startTok.getLoc());

  SmallVector<Attribute> tags;
  SmallVector<std::unique_ptr<Region>> regions;
  while (true) {
    auto tagLoc = getToken().getLoc();

    // Only consume the keyword if the indentation is correct.
    auto caseIndent = getIndentation();
    if (!caseIndent || *caseIndent <= matchIndent)
      break;

    // Parse the tag.
    StringRef tagSpelling;
    if (parseId(tagSpelling, "expected enumeration tag in match statement"))
      return failure();
    auto tagIndex = enumType.getElementIndex(tagSpelling);
    if (!tagIndex)
      return emitError(tagLoc, "tag ")
             << tagSpelling << " not a member of enumeration " << enumType;
    auto tag = IntegerAttr::get(IntegerType::get(getContext(), 32), *tagIndex);
    tags.push_back(tag);

    // Add a new case to the match operation.
    auto *caseBlock = &regions.emplace_back(new Region)->emplaceBlock();

    // Declarations are scoped to the case.
    FIRModuleContext::ContextScope scope(moduleContext, caseBlock);

    // After parsing the region, we can release any new entries in
    // unbundledValues since the symbol table entries that refer to them will be
    // gone.
    UnbundledValueRestorer x(moduleContext.unbundledValues);

    // Parse the argument.
    if (consumeIf(FIRToken::l_paren)) {
      StringAttr identifier;
      if (parseId(identifier, "expected identifier for 'case' binding"))
        return failure();

      // Add an argument to the block.
      auto dataType = enumType.getElementType(*tagIndex);
      caseBlock->addArgument(dataType, LocWithInfo(tagLoc, this).getLoc());

      if (moduleContext.addSymbolEntry(identifier, caseBlock->getArgument(0),
                                       startTok.getLoc()))
        return failure();

      if (parseToken(FIRToken::r_paren, "expected ')' in match statement case"))
        return failure();

    } else {
      auto dataType = IntType::get(builder.getContext(), false, 0);
      caseBlock->addArgument(dataType, LocWithInfo(tagLoc, this).getLoc());
    }

    if (parseToken(FIRToken::colon, "expected ':' in match statement case"))
      return failure();

    // Parse a block of statements that are indented more than the case.
    auto subParser = std::make_unique<FIRStmtParser>(
        *caseBlock, moduleContext, innerSymFixups, circuitSymTbl, version,
        layerSym);
    if (subParser->parseSimpleStmtBlock(*caseIndent))
      return failure();
  }

  MatchOp::create(builder, input, ArrayAttr::get(getContext(), tags), regions);
  return success();
}

/// ref_expr ::= probe | rwprobe | static_reference
// NOLINTNEXTLINE(misc-no-recursion)
ParseResult FIRStmtParser::parseRefExp(Value &result, const Twine &message) {
  auto token = getToken().getKind();
  if (token == FIRToken::lp_probe)
    return parseProbe(result);
  if (token == FIRToken::lp_rwprobe)
    return parseRWProbe(result);

  // Default to parsing as static reference expression.
  // Don't check token kind, we need to support literal_identifier and keywords,
  // let parseId handle this.
  return parseStaticRefExp(result, message);
}

/// static_reference ::= id
///                  ::= static_reference '.' id
///                  ::= static_reference '[' int ']'
// NOLINTNEXTLINE(misc-no-recursion)
ParseResult FIRStmtParser::parseStaticRefExp(Value &result,
                                             const Twine &message) {
  auto parseIdOrInstance = [&]() -> ParseResult {
    StringRef id;
    auto loc = getToken().getLoc();
    SymbolValueEntry symtabEntry;
    if (parseId(id, message) ||
        moduleContext.lookupSymbolEntry(symtabEntry, id, loc))
      return failure();

    // If we looked up a normal value, then we're done.
    if (!moduleContext.resolveSymbolEntry(result, symtabEntry, loc, false))
      return success();

    assert(isa<UnbundledID>(symtabEntry) && "should be an instance");

    // Handle the normal "instance.x" reference.
    StringRef fieldName;
    return failure(
        parseToken(FIRToken::period, "expected '.' in field reference") ||
        parseFieldId(fieldName, "expected field name") ||
        moduleContext.resolveSymbolEntry(result, symtabEntry, fieldName, loc));
  };
  return failure(parseIdOrInstance() ||
                 parseOptionalExpPostscript(result, false));
}
/// static_reference ::= id
///                  ::= static_reference '.' id
///                  ::= static_reference '[' int ']'
/// Populate `refResult` with rwprobe "root" and parsed indexing.
/// Root is base-type target, and will be block argument or forceable.
/// Also set `Type`, so we can handle const-ness while visiting.
/// If root is an unbundled entry, replace with bounce wire and update
/// the unbundled entry to point to this for future users.
// NOLINTNEXTLINE(misc-no-recursion)
ParseResult FIRStmtParser::parseRWProbeStaticRefExp(FieldRef &refResult,
                                                    Type &type,
                                                    const Twine &message) {
  auto loc = getToken().getLoc();

  StringRef id;
  SymbolValueEntry symtabEntry;
  if (parseId(id, message) ||
      moduleContext.lookupSymbolEntry(symtabEntry, id, loc))
    return failure();

  // Three kinds of rwprobe targets:
  // 1. Instance result.  Replace with a forceable wire, handle as (2).
  // 2. Forceable declaration.
  // 3. BlockArgument.

  // We use inner symbols for all.

  // Figure out what we have, and parse indexing.
  Value result;
  if (auto unbundledId = dyn_cast<UnbundledID>(symtabEntry)) {
    // This means we have an instance.
    auto &ubEntry = moduleContext.getUnbundledEntry(unbundledId - 1);

    StringRef fieldName;
    auto loc = getToken().getLoc();
    if (parseToken(FIRToken::period, "expected '.' in field reference") ||
        parseFieldId(fieldName, "expected field name"))
      return failure();

    // Find unbundled entry for the specified result/port.
    // Get a reference to it--as we may update it (!!).
    auto fieldAttr = StringAttr::get(getContext(), fieldName);
    for (auto &elt : ubEntry) {
      if (elt.first == fieldAttr) {
        // Grab the unbundled entry /by reference/ so we can update it with the
        // new forceable wire we insert (if not already done).
        auto &instResult = elt.second;

        // If it's already forceable, use that.
        auto *defining = instResult.getDefiningOp();
        assert(defining);
        if (isa<WireOp>(defining)) {
          result = instResult;
          break;
        }

        // Otherwise, replace with bounce wire.
        auto type = instResult.getType();

        // Either entire instance result is forceable + bounce wire, or reject.
        // (even if rwprobe is of a portion of the port)
        bool forceable = static_cast<bool>(
            firrtl::detail::getForceableResultType(true, type));
        if (!forceable)
          return emitError(loc, "unable to force instance result of type ")
                 << type;

        // Create bounce wire for the instance result.
        auto annotations = getConstants().emptyArrayAttr;
        StringAttr sym = {};
        SmallString<64> name;
        (id + "_" + fieldName + "_bounce").toVector(name);
        locationProcessor.setLoc(loc);
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(defining);
        auto bounce =
            WireOp::create(builder, type, name, NameKindEnum::InterestingName,
                           annotations, sym);
        auto bounceVal = bounce.getData();

        // Replace instance result with reads from bounce wire.
        instResult.replaceAllUsesWith(bounceVal);

        // Connect to/from the result per flow.
        builder.setInsertionPointAfter(defining);
        if (foldFlow(instResult) == Flow::Source)
          emitConnect(builder, bounceVal, instResult);
        else
          emitConnect(builder, instResult, bounceVal);
        // Set the parse result AND update `instResult` which is a reference to
        // the unbundled entry for the instance result, so that future uses also
        // find this new wire.
        result = instResult = bounce.getDataRaw();
        break;
      }
    }

    if (!result) {
      emitError(loc, "use of invalid field name '")
          << fieldName << "' on bundle value";
      return failure();
    }
  } else {
    // This target can be a port or a regular value.
    result = cast<Value>(symtabEntry);
  }

  assert(result);
  assert(isa<BlockArgument>(result) ||
         result.getDefiningOp<hw::InnerSymbolOpInterface>());

  // We have our root value, we just need to parse the field id.
  // Build up the FieldRef as processing indexing expressions, and
  // compute the type so that we know the const-ness of the final expression.
  refResult = FieldRef(result, 0);
  type = result.getType();
  while (true) {
    if (consumeIf(FIRToken::period)) {
      SmallVector<StringRef, 3> fields;
      if (parseFieldIdSeq(fields, "expected field name"))
        return failure();
      for (auto fieldName : fields) {
        if (auto bundle = type_dyn_cast<BundleType>(type)) {
          if (auto index = bundle.getElementIndex(fieldName)) {
            refResult = refResult.getSubField(bundle.getFieldID(*index));
            type = bundle.getElementTypePreservingConst(*index);
            continue;
          }
        } else if (auto bundle = type_dyn_cast<OpenBundleType>(type)) {
          if (auto index = bundle.getElementIndex(fieldName)) {
            refResult = refResult.getSubField(bundle.getFieldID(*index));
            type = bundle.getElementTypePreservingConst(*index);
            continue;
          }
        } else {
          return emitError(loc, "subfield requires bundle operand")
                 << "got " << type << "\n";
        }
        return emitError(loc,
                         "unknown field '" + fieldName + "' in bundle type ")
               << type;
      }
      continue;
    }
    if (consumeIf(FIRToken::l_square)) {
      auto loc = getToken().getLoc();
      int32_t index;
      if (parseIntLit(index, "expected index") ||
          parseToken(FIRToken::r_square, "expected ']'"))
        return failure();

      if (index < 0)
        return emitError(loc, "invalid index specifier");

      if (auto vector = type_dyn_cast<FVectorType>(type)) {
        if ((unsigned)index < vector.getNumElements()) {
          refResult = refResult.getSubField(vector.getFieldID(index));
          type = vector.getElementTypePreservingConst();
          continue;
        }
      } else if (auto vector = type_dyn_cast<OpenVectorType>(type)) {
        if ((unsigned)index < vector.getNumElements()) {
          refResult = refResult.getSubField(vector.getFieldID(index));
          type = vector.getElementTypePreservingConst();
          continue;
        }
      } else {
        return emitError(loc, "subindex requires vector operand");
      }
      return emitError(loc, "out of range index '")
             << index << "' for vector type " << type;
    }
    return success();
  }
}

/// intrinsic_expr ::=  'intrinsic(' Id (params)?  ':' type    exp* ')'
/// intrinsic_stmt ::=  'intrinsic(' Id (params)? (':' type )? exp* ')'
ParseResult FIRStmtParser::parseIntrinsic(Value &result, bool isStatement) {
  auto startTok = consumeToken(FIRToken::lp_intrinsic);
  StringRef intrinsic;
  ArrayAttr parameters;
  FIRRTLType type;

  if (parseId(intrinsic, "expected intrinsic identifier") ||
      parseOptionalParams(parameters))
    return failure();

  if (consumeIf(FIRToken::colon)) {
    if (parseType(type, "expected intrinsic return type"))
      return failure();
  } else if (!isStatement)
    return emitError("expected ':' in intrinsic expression");

  SmallVector<Value> operands;
  auto loc = startTok.getLoc();
  if (consumeIf(FIRToken::comma)) {
    if (parseListUntil(FIRToken::r_paren, [&]() -> ParseResult {
          Value operand;
          if (parseExp(operand, "expected operand in intrinsic"))
            return failure();
          operands.push_back(operand);
          locationProcessor.setLoc(loc);
          return success();
        }))
      return failure();
  } else {
    if (parseToken(FIRToken::r_paren, "expected ')' in intrinsic"))
      return failure();
  }

  if (isStatement)
    if (parseOptionalInfo())
      return failure();

  locationProcessor.setLoc(loc);

  auto op = GenericIntrinsicOp::create(
      builder, type, builder.getStringAttr(intrinsic), operands, parameters);
  if (type)
    result = op.getResult();
  return success();
}

/// params ::= '<' param','* '>'
ParseResult FIRStmtParser::parseOptionalParams(ArrayAttr &resultParameters) {
  if (!consumeIf(FIRToken::less))
    return success();

  SmallVector<Attribute, 8> parameters;
  SmallPtrSet<StringAttr, 8> seen;
  if (parseListUntil(FIRToken::greater, [&]() -> ParseResult {
        StringAttr name;
        Attribute value;
        SMLoc loc;
        if (parseParameter(name, value, loc))
          return failure();
        auto typedValue = dyn_cast<TypedAttr>(value);
        if (!typedValue)
          return emitError(loc)
                 << "invalid value for parameter '" << name.getValue() << "'";
        if (!seen.insert(name).second)
          return emitError(loc, "redefinition of parameter '" +
                                    name.getValue() + "'");
        parameters.push_back(ParamDeclAttr::get(name, typedValue));
        return success();
      }))
    return failure();

  resultParameters = ArrayAttr::get(getContext(), parameters);
  return success();
}

/// path ::= 'path(' StringLit ')'
// NOLINTNEXTLINE(misc-no-recursion)
ParseResult FIRStmtParser::parsePathExp(Value &result) {
  auto startTok = consumeToken(FIRToken::lp_path);
  locationProcessor.setLoc(startTok.getLoc());
  StringRef target;
  if (parseGetSpelling(target) ||
      parseToken(FIRToken::string,
                 "expected target string in path expression") ||
      parseToken(FIRToken::r_paren, "expected ')' in path expression"))
    return failure();
  result = UnresolvedPathOp::create(
      builder, StringAttr::get(getContext(), FIRToken::getStringValue(target)));
  return success();
}

/// define ::= 'define' static_reference '=' ref_expr info?
ParseResult FIRStmtParser::parseRefDefine() {
  auto startTok = consumeToken(FIRToken::kw_define);

  Value src, target;
  if (parseStaticRefExp(target,
                        "expected static reference expression in 'define'") ||
      parseToken(FIRToken::equal,
                 "expected '=' after define reference expression") ||
      parseRefExp(src, "expected reference expression in 'define'") ||
      parseOptionalInfo())
    return failure();

  // Check reference expressions are of reference type.
  if (!type_isa<RefType>(target.getType()))
    return emitError(startTok.getLoc(), "expected reference-type expression in "
                                        "'define' target (LHS), got ")
           << target.getType();
  if (!type_isa<RefType>(src.getType()))
    return emitError(startTok.getLoc(), "expected reference-type expression in "
                                        "'define' source (RHS), got ")
           << src.getType();

  // static_reference doesn't differentiate which can be ref.sub'd, so check
  // this explicitly:
  if (isa_and_nonnull<RefSubOp>(target.getDefiningOp()))
    return emitError(startTok.getLoc(),
                     "cannot define into a sub-element of a reference");

  locationProcessor.setLoc(startTok.getLoc());

  if (!areTypesRefCastable(target.getType(), src.getType()))
    return emitError(startTok.getLoc(), "cannot define reference of type ")
           << target.getType() << " with incompatible reference of type "
           << src.getType();

  emitConnect(builder, target, src);

  return success();
}

/// read ::= '(' ref_expr ')'
/// XXX: spec says static_reference, allow ref_expr anyway for read(probe(x)).
ParseResult FIRStmtParser::parseRefRead(Value &result) {
  auto startTok = consumeToken(FIRToken::lp_read);

  Value ref;
  if (parseRefExp(ref, "expected reference expression in 'read'") ||
      parseToken(FIRToken::r_paren, "expected ')' in 'read'"))
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  // Check argument is a ref-type value.
  if (!type_isa<RefType>(ref.getType()))
    return emitError(startTok.getLoc(),
                     "expected reference-type expression in 'read', got ")
           << ref.getType();

  result = RefResolveOp::create(builder, ref);

  return success();
}

/// probe ::= 'probe' '(' static_ref ')'
ParseResult FIRStmtParser::parseProbe(Value &result) {
  auto startTok = consumeToken(FIRToken::lp_probe);

  Value staticRef;
  if (parseStaticRefExp(staticRef,
                        "expected static reference expression in 'probe'") ||
      parseToken(FIRToken::r_paren, "expected ')' in 'probe'"))
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  // Check probe expression is base-type.
  if (!type_isa<FIRRTLBaseType>(staticRef.getType()))
    return emitError(startTok.getLoc(),
                     "expected base-type expression in 'probe', got ")
           << staticRef.getType();

  // Check for other unsupported reference sources.
  // TODO: Add to ref.send verifier / inferReturnTypes.
  if (isa_and_nonnull<MemOp, CombMemOp, SeqMemOp, MemoryPortOp,
                      MemoryDebugPortOp, MemoryPortAccessOp>(
          staticRef.getDefiningOp()))
    return emitError(startTok.getLoc(), "cannot probe memories or their ports");

  result = RefSendOp::create(builder, staticRef);

  return success();
}

/// rwprobe ::= 'rwprobe' '(' static_ref ')'
ParseResult FIRStmtParser::parseRWProbe(Value &result) {
  auto startTok = consumeToken(FIRToken::lp_rwprobe);

  FieldRef staticRef;
  Type parsedTargetType;
  if (parseRWProbeStaticRefExp(
          staticRef, parsedTargetType,
          "expected static reference expression in 'rwprobe'") ||
      parseToken(FIRToken::r_paren, "expected ')' in 'rwprobe'"))
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  // Checks:
  // Not public port (verifier)

  // Check probe expression is base-type.
  auto targetType = type_dyn_cast<FIRRTLBaseType>(parsedTargetType);
  if (!targetType)
    return emitError(startTok.getLoc(),
                     "expected base-type expression in 'rwprobe', got ")
           << parsedTargetType;

  auto root = staticRef.getValue();
  auto *definingOp = root.getDefiningOp();

  if (isa_and_nonnull<MemOp, CombMemOp, SeqMemOp, MemoryPortOp,
                      MemoryDebugPortOp, MemoryPortAccessOp>(definingOp))
    return emitError(startTok.getLoc(), "cannot probe memories or their ports");

  auto forceableType = firrtl::detail::getForceableResultType(true, targetType);
  if (!forceableType)
    return emitError(startTok.getLoc(), "cannot force target of type ")
           << targetType;

  // Create the operation with a placeholder reference and add to fixup list.
  auto op = RWProbeOp::create(builder, forceableType,
                              getConstants().placeholderInnerRef);
  innerSymFixups.add(op, getTargetFor(staticRef));
  result = op;
  return success();
}

/// force ::= 'force(' exp exp ref_expr exp ')' info?
ParseResult FIRStmtParser::parseRefForce() {
  auto startTok = consumeToken(FIRToken::lp_force);

  Value clock, pred, dest, src;
  if (parseExp(clock, "expected clock expression in force") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(pred, "expected predicate expression in force") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseRefExp(dest, "expected destination reference expression in force") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(src, "expected source expression in force") ||
      parseToken(FIRToken::r_paren, "expected ')' in force") ||
      parseOptionalInfo())
    return failure();

  // Check reference expression is of reference type.
  auto ref = type_dyn_cast<RefType>(dest.getType());
  if (!ref || !ref.getForceable())
    return emitError(
               startTok.getLoc(),
               "expected rwprobe-type expression for force destination, got ")
           << dest.getType();
  auto srcBaseType = type_dyn_cast<FIRRTLBaseType>(src.getType());
  if (!srcBaseType)
    return emitError(startTok.getLoc(),
                     "expected base-type for force source, got ")
           << src.getType();
  if (!srcBaseType.isPassive())
    return emitError(startTok.getLoc(),
                     "expected passive value for force source, got ")
           << srcBaseType;

  locationProcessor.setLoc(startTok.getLoc());

  // Cast ref to accommodate uninferred sources.
  auto noConstSrcType = srcBaseType.getAllConstDroppedType();
  if (noConstSrcType != ref.getType()) {
    // Try to cast destination to rwprobe of source type (dropping const).
    auto compatibleRWProbe = RefType::get(noConstSrcType, true, ref.getLayer());
    if (areTypesRefCastable(compatibleRWProbe, ref))
      dest = RefCastOp::create(builder, compatibleRWProbe, dest);
    else
      return emitError(startTok.getLoc(), "incompatible force source of type ")
             << src.getType() << " cannot target destination "
             << dest.getType();
  }

  RefForceOp::create(builder, clock, pred, dest, src);

  return success();
}

/// force_initial ::= 'force_initial(' ref_expr exp ')' info?
ParseResult FIRStmtParser::parseRefForceInitial() {
  auto startTok = consumeToken(FIRToken::lp_force_initial);

  Value dest, src;
  if (parseRefExp(
          dest, "expected destination reference expression in force_initial") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(src, "expected source expression in force_initial") ||
      parseToken(FIRToken::r_paren, "expected ')' in force_initial") ||
      parseOptionalInfo())
    return failure();

  // Check reference expression is of reference type.
  auto ref = type_dyn_cast<RefType>(dest.getType());
  if (!ref || !ref.getForceable())
    return emitError(startTok.getLoc(), "expected rwprobe-type expression for "
                                        "force_initial destination, got ")
           << dest.getType();
  auto srcBaseType = type_dyn_cast<FIRRTLBaseType>(src.getType());
  if (!srcBaseType)
    return emitError(startTok.getLoc(),
                     "expected base-type expression for force_initial "
                     "source, got ")
           << src.getType();
  if (!srcBaseType.isPassive())
    return emitError(startTok.getLoc(),
                     "expected passive value for force_initial source, got ")
           << srcBaseType;

  locationProcessor.setLoc(startTok.getLoc());

  // Cast ref to accommodate uninferred sources.
  auto noConstSrcType = srcBaseType.getAllConstDroppedType();
  if (noConstSrcType != ref.getType()) {
    // Try to cast destination to rwprobe of source type (dropping const).
    auto compatibleRWProbe = RefType::get(noConstSrcType, true, ref.getLayer());
    if (areTypesRefCastable(compatibleRWProbe, ref))
      dest = RefCastOp::create(builder, compatibleRWProbe, dest);
    else
      return emitError(startTok.getLoc(),
                       "incompatible force_initial source of type ")
             << src.getType() << " cannot target destination "
             << dest.getType();
  }

  auto value = APInt::getAllOnes(1);
  auto type = UIntType::get(builder.getContext(), 1);
  auto attr = builder.getIntegerAttr(IntegerType::get(type.getContext(),
                                                      value.getBitWidth(),
                                                      IntegerType::Unsigned),
                                     value);
  auto pred = moduleContext.getCachedConstant(builder, attr, type, attr);
  RefForceInitialOp::create(builder, pred, dest, src);

  return success();
}

/// release ::= 'release(' exp exp ref_expr ')' info?
ParseResult FIRStmtParser::parseRefRelease() {
  auto startTok = consumeToken(FIRToken::lp_release);

  Value clock, pred, dest;
  if (parseExp(clock, "expected clock expression in release") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(pred, "expected predicate expression in release") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseRefExp(dest,
                  "expected destination reference expression in release") ||
      parseToken(FIRToken::r_paren, "expected ')' in release") ||
      parseOptionalInfo())
    return failure();

  // Check reference expression is of reference type.
  if (auto ref = type_dyn_cast<RefType>(dest.getType());
      !ref || !ref.getForceable())
    return emitError(
               startTok.getLoc(),
               "expected rwprobe-type expression for release destination, got ")
           << dest.getType();

  locationProcessor.setLoc(startTok.getLoc());

  RefReleaseOp::create(builder, clock, pred, dest);

  return success();
}

/// release_initial ::= 'release_initial(' ref_expr ')' info?
ParseResult FIRStmtParser::parseRefReleaseInitial() {
  auto startTok = consumeToken(FIRToken::lp_release_initial);

  Value dest;
  if (parseRefExp(
          dest,
          "expected destination reference expression in release_initial") ||
      parseToken(FIRToken::r_paren, "expected ')' in release_initial") ||
      parseOptionalInfo())
    return failure();

  // Check reference expression is of reference type.
  if (auto ref = type_dyn_cast<RefType>(dest.getType());
      !ref || !ref.getForceable())
    return emitError(startTok.getLoc(), "expected rwprobe-type expression for "
                                        "release_initial destination, got ")
           << dest.getType();

  locationProcessor.setLoc(startTok.getLoc());

  auto value = APInt::getAllOnes(1);
  auto type = UIntType::get(builder.getContext(), 1);
  auto attr = builder.getIntegerAttr(IntegerType::get(type.getContext(),
                                                      value.getBitWidth(),
                                                      IntegerType::Unsigned),
                                     value);
  auto pred = moduleContext.getCachedConstant(builder, attr, type, attr);
  RefReleaseInitialOp::create(builder, pred, dest);

  return success();
}

/// connect ::= 'connect' expr expr
ParseResult FIRStmtParser::parseConnect() {
  auto startTok = consumeToken(FIRToken::kw_connect);
  auto loc = startTok.getLoc();

  Value lhs, rhs;
  if (parseExp(lhs, "expected connect expression") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(rhs, "expected connect expression") || parseOptionalInfo())
    return failure();

  auto lhsType = type_dyn_cast<FIRRTLBaseType>(lhs.getType());
  auto rhsType = type_dyn_cast<FIRRTLBaseType>(rhs.getType());
  if (!lhsType || !rhsType)
    return emitError(loc, "cannot connect reference or property types");
  // TODO: Once support lands for agg-of-ref, add test for this check!
  if (lhsType.containsReference() || rhsType.containsReference())
    return emitError(loc, "cannot connect types containing references");

  if (!areTypesEquivalent(lhsType, rhsType))
    return emitError(loc, "cannot connect non-equivalent type ")
           << rhsType << " to " << lhsType;

  locationProcessor.setLoc(loc);
  emitConnect(builder, lhs, rhs);
  return success();
}

/// propassign ::= 'propassign' expr expr
ParseResult FIRStmtParser::parsePropAssign() {
  auto startTok = consumeToken(FIRToken::kw_propassign);
  auto loc = startTok.getLoc();

  Value lhs, rhs;
  if (parseExp(lhs, "expected propassign expression") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(rhs, "expected propassign expression") || parseOptionalInfo())
    return failure();

  auto lhsType = type_dyn_cast<PropertyType>(lhs.getType());
  auto rhsType = type_dyn_cast<PropertyType>(rhs.getType());
  if (!lhsType || !rhsType)
    return emitError(loc, "can only propassign property types");
  locationProcessor.setLoc(loc);
  if (lhsType != rhsType) {
    // If the lhs is anyref, and the rhs is a ClassType, insert a cast.
    if (isa<AnyRefType>(lhsType) && isa<ClassType>(rhsType))
      rhs = ObjectAnyRefCastOp::create(builder, rhs);
    else
      return emitError(loc, "cannot propassign non-equivalent type ")
             << rhsType << " to " << lhsType;
  }
  PropAssignOp::create(builder, lhs, rhs);
  return success();
}

/// invalidate ::= 'invalidate' expr
ParseResult FIRStmtParser::parseInvalidate() {
  auto startTok = consumeToken(FIRToken::kw_invalidate);

  Value lhs;

  StringRef id;
  auto loc = getToken().getLoc();
  SymbolValueEntry symtabEntry;
  if (parseId(id, "expected static reference expression") ||
      moduleContext.lookupSymbolEntry(symtabEntry, id, loc))
    return failure();

  // If we looked up a normal value (e.g., wire, register, or port), then we
  // just need to get any optional trailing expression.  Invalidate this.
  if (!moduleContext.resolveSymbolEntry(lhs, symtabEntry, loc, false)) {
    if (parseOptionalExpPostscript(lhs, /*allowDynamic=*/false) ||
        parseOptionalInfo())
      return failure();

    locationProcessor.setLoc(startTok.getLoc());
    emitInvalidate(lhs);
    return success();
  }

  // We're dealing with an instance.  This instance may or may not have a
  // trailing expression.  Handle the special case of no trailing expression
  // first by invalidating all of its results.
  assert(isa<UnbundledID>(symtabEntry) && "should be an instance");

  if (getToken().isNot(FIRToken::period)) {
    locationProcessor.setLoc(loc);
    // Invalidate all of the results of the bundled value.
    unsigned unbundledId = cast<UnbundledID>(symtabEntry) - 1;
    UnbundledValueEntry &ubEntry = moduleContext.getUnbundledEntry(unbundledId);
    for (auto elt : ubEntry)
      emitInvalidate(elt.second);
    return success();
  }

  // Handle the case of an instance with a trailing expression.  This must begin
  // with a '.' (until we add instance arrays).
  StringRef fieldName;
  if (parseToken(FIRToken::period, "expected '.' in field reference") ||
      parseFieldId(fieldName, "expected field name") ||
      moduleContext.resolveSymbolEntry(lhs, symtabEntry, fieldName, loc))
    return failure();

  // Update with any trailing expression and invalidate it.
  if (parseOptionalExpPostscript(lhs, /*allowDynamic=*/false) ||
      parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());
  emitInvalidate(lhs);
  return success();
}

ParseResult FIRStmtParser::parseLayerBlockOrGroup(unsigned indent) {

  auto startTok = consumeToken();
  assert(startTok.isAny(FIRToken::kw_layerblock, FIRToken::kw_group) &&
         "consumed an unexpected token");
  auto loc = startTok.getLoc();

  StringRef id;
  if (parseId(id, "expected layer identifer") ||
      parseToken(FIRToken::colon, "expected ':' at end of layer block") ||
      parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(loc);

  StringRef rootLayer;
  SmallVector<FlatSymbolRefAttr> nestedLayers;
  if (!layerSym) {
    rootLayer = id;
  } else {
    rootLayer = layerSym.getRootReference();
    auto nestedRefs = layerSym.getNestedReferences();
    nestedLayers.append(nestedRefs.begin(), nestedRefs.end());
    nestedLayers.push_back(FlatSymbolRefAttr::get(builder.getContext(), id));
  }

  auto layerBlockOp = LayerBlockOp::create(
      builder,
      SymbolRefAttr::get(builder.getContext(), rootLayer, nestedLayers));
  layerBlockOp->getRegion(0).push_back(new Block());

  if (getIndentation() > indent)
    if (parseSubBlock(layerBlockOp.getRegion().front(), indent,
                      layerBlockOp.getLayerName()))
      return failure();

  return success();
}

/// leading-exp-stmt ::= exp '<=' exp info?
///                  ::= exp 'is' 'invalid' info?
ParseResult FIRStmtParser::parseLeadingExpStmt(Value lhs) {
  auto loc = getToken().getLoc();

  // If 'is' grammar is special.
  if (consumeIf(FIRToken::kw_is)) {
    if (parseToken(FIRToken::kw_invalid, "expected 'invalid'") ||
        parseOptionalInfo())
      return failure();

    if (removedFeature({3, 0, 0}, "'is invalid' statements", loc))
      return failure();

    locationProcessor.setLoc(loc);
    emitInvalidate(lhs);
    return success();
  }

  if (parseToken(FIRToken::less_equal, "expected '<=' in statement"))
    return failure();

  if (removedFeature({3, 0, 0}, "'<=' connections", loc))
    return failure();

  Value rhs;
  if (parseExp(rhs, "unexpected token in statement") || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(loc);

  auto lhsType = type_dyn_cast<FIRRTLBaseType>(lhs.getType());
  auto rhsType = type_dyn_cast<FIRRTLBaseType>(rhs.getType());
  if (!lhsType || !rhsType)
    return emitError(loc, "cannot connect reference or property types");
  // TODO: Once support lands for agg-of-ref, add test for this check!
  if (lhsType.containsReference() || rhsType.containsReference())
    return emitError(loc, "cannot connect types containing references");

  if (!areTypesEquivalent(lhsType, rhsType))
    return emitError(loc, "cannot connect non-equivalent type ")
           << rhsType << " to " << lhsType;
  emitConnect(builder, lhs, rhs);
  return success();
}

//===-------------------------------
// FIRStmtParser Declaration Parsing

/// instance ::= 'inst' id 'of' id info?
ParseResult FIRStmtParser::parseInstance() {
  auto startTok = consumeToken(FIRToken::kw_inst);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  StringRef id;
  StringRef moduleName;
  if (parseId(id, "expected instance name") ||
      parseToken(FIRToken::kw_of, "expected 'of' in instance") ||
      parseId(moduleName, "expected module name") || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  // Look up the module that is being referenced.
  auto referencedModule = getReferencedModule(startTok.getLoc(), moduleName);
  if (!referencedModule)
    return failure();

  SmallVector<PortInfo> modulePorts = referencedModule.getPorts();

  auto annotations = getConstants().emptyArrayAttr;
  SmallVector<Attribute, 4> portAnnotations(modulePorts.size(), annotations);

  hw::InnerSymAttr sym = {};
  auto result = InstanceOp::create(
      builder, referencedModule, id, NameKindEnum::InterestingName,
      annotations.getValue(), portAnnotations, false, false, sym);

  // Since we are implicitly unbundling the instance results, we need to keep
  // track of the mapping from bundle fields to results in the unbundledValues
  // data structure.  Build our entry now.
  UnbundledValueEntry unbundledValueEntry;
  unbundledValueEntry.reserve(modulePorts.size());
  for (size_t i = 0, e = modulePorts.size(); i != e; ++i)
    unbundledValueEntry.push_back({modulePorts[i].name, result.getResult(i)});

  // Add it to unbundledValues and add an entry to the symbol table to remember
  // it.
  moduleContext.unbundledValues.push_back(std::move(unbundledValueEntry));
  auto entryId = UnbundledID(moduleContext.unbundledValues.size());
  return moduleContext.addSymbolEntry(id, entryId, startTok.getLoc());
}

/// instance_choice ::=
///   'inst_choice' id 'of' id id info? newline indent ( id "=>" id )+ dedent
ParseResult FIRStmtParser::parseInstanceChoice() {
  auto startTok = consumeToken(FIRToken::kw_instchoice);
  SMLoc loc = startTok.getLoc();

  // If this was actually the start of a connect or something else handle that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  if (requireFeature(missingSpecFIRVersion, "option groups/instance choices"))
    return failure();

  StringRef id;
  StringRef defaultModuleName;
  StringRef optionGroupName;
  if (parseId(id, "expected instance name") ||
      parseToken(FIRToken::kw_of, "expected 'of' in instance") ||
      parseId(defaultModuleName, "expected module name") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseId(optionGroupName, "expected option group name") ||
      parseToken(FIRToken::colon, "expected ':' after instchoice") ||
      parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  // Look up the default module referenced by the instance choice.
  // The port lists of all the other referenced modules must match this one.
  auto defaultModule = getReferencedModule(loc, defaultModuleName);
  if (!defaultModule)
    return failure();

  SmallVector<PortInfo> modulePorts = defaultModule.getPorts();

  // Find the option group.
  auto optionGroup = circuitSymTbl.lookup<OptionOp>(optionGroupName);
  if (!optionGroup)
    return emitError(loc,
                     "use of undefined option group '" + optionGroupName + "'");

  auto baseIndent = getIndentation();
  SmallVector<std::pair<OptionCaseOp, FModuleLike>> caseModules;
  while (getIndentation() == baseIndent) {
    StringRef caseId;
    StringRef caseModuleName;
    if (parseId(caseId, "expected a case identifier") ||
        parseToken(FIRToken::equal_greater,
                   "expected '=> in instance choice definition") ||
        parseId(caseModuleName, "expected module name"))
      return failure();

    auto caseModule = getReferencedModule(loc, caseModuleName);
    if (!caseModule)
      return failure();

    for (const auto &[defaultPort, casePort] :
         llvm::zip(modulePorts, caseModule.getPorts())) {
      if (defaultPort.name != casePort.name)
        return emitError(loc, "instance case module port '")
               << casePort.name.getValue()
               << "' does not match the default module port '"
               << defaultPort.name.getValue() << "'";
      if (defaultPort.type != casePort.type)
        return emitError(loc, "instance case port '")
               << casePort.name.getValue()
               << "' type does not match the default module port";
    }

    auto optionCase =
        dyn_cast_or_null<OptionCaseOp>(optionGroup.lookupSymbol(caseId));
    if (!optionCase)
      return emitError(loc, "use of undefined option case '" + caseId + "'");
    caseModules.emplace_back(optionCase, caseModule);
  }

  auto annotations = getConstants().emptyArrayAttr;
  SmallVector<Attribute, 4> portAnnotations(modulePorts.size(), annotations);

  // Create an instance choice op.
  StringAttr sym;
  auto result = InstanceChoiceOp::create(
      builder, defaultModule, caseModules, id, NameKindEnum::InterestingName,
      annotations.getValue(), portAnnotations, sym);

  // Un-bundle the ports, identically to the regular instance operation.
  UnbundledValueEntry unbundledValueEntry;
  unbundledValueEntry.reserve(modulePorts.size());
  for (size_t i = 0, e = modulePorts.size(); i != e; ++i)
    unbundledValueEntry.push_back({modulePorts[i].name, result.getResult(i)});

  moduleContext.unbundledValues.push_back(std::move(unbundledValueEntry));
  auto entryId = UnbundledID(moduleContext.unbundledValues.size());
  return moduleContext.addSymbolEntry(id, entryId, startTok.getLoc());
}

FModuleLike FIRStmtParser::getReferencedModule(SMLoc loc,
                                               StringRef moduleName) {
  auto referencedModule = circuitSymTbl.lookup<FModuleLike>(moduleName);
  if (!referencedModule) {
    emitError(loc,
              "use of undefined module name '" + moduleName + "' in instance");
    return {};
  }
  if (isa<ClassOp /* ClassLike */>(referencedModule)) {
    emitError(loc, "cannot create instance of class '" + moduleName +
                       "', did you mean object?");
    return {};
  }
  return referencedModule;
}

/// object ::= 'object' id 'of' id info?
ParseResult FIRStmtParser::parseObject() {
  auto startTok = consumeToken(FIRToken::kw_object);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  if (requireFeature(missingSpecFIRVersion, "object statements"))
    return failure();

  StringRef id;
  StringRef className;
  if (parseId(id, "expected object name") ||
      parseToken(FIRToken::kw_of, "expected 'of' in object") ||
      parseId(className, "expected class name") || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  // Look up the class that is being referenced.
  const auto &classMap = getConstants().classMap;
  auto lookup = classMap.find(className);
  if (lookup == classMap.end())
    return emitError(startTok.getLoc(), "use of undefined class name '" +
                                            className + "' in object");
  auto referencedClass = lookup->getSecond();
  auto result = ObjectOp::create(builder, referencedClass, id);
  return moduleContext.addSymbolEntry(id, result, startTok.getLoc());
}

/// cmem ::= 'cmem' id ':' type info?
ParseResult FIRStmtParser::parseCombMem() {
  // TODO(firrtl spec) cmem is completely undocumented.
  auto startTok = consumeToken(FIRToken::kw_cmem);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  StringRef id;
  FIRRTLType type;
  if (parseId(id, "expected cmem name") ||
      parseToken(FIRToken::colon, "expected ':' in cmem") ||
      parseType(type, "expected cmem type") || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  // Transform the parsed vector type into a memory type.
  auto vectorType = type_dyn_cast<FVectorType>(type);
  if (!vectorType)
    return emitError("cmem requires vector type");

  auto annotations = getConstants().emptyArrayAttr;
  StringAttr sym = {};
  auto result = CombMemOp::create(
      builder, vectorType.getElementType(), vectorType.getNumElements(), id,
      NameKindEnum::InterestingName, annotations, sym);
  return moduleContext.addSymbolEntry(id, result, startTok.getLoc());
}

/// smem ::= 'smem' id ':' type ruw? info?
ParseResult FIRStmtParser::parseSeqMem() {
  // TODO(firrtl spec) smem is completely undocumented.
  auto startTok = consumeToken(FIRToken::kw_smem);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  StringRef id;
  FIRRTLType type;
  RUWBehavior ruw = RUWBehavior::Undefined;

  if (parseId(id, "expected smem name") ||
      parseToken(FIRToken::colon, "expected ':' in smem") ||
      parseType(type, "expected smem type"))
    return failure();

  if (consumeIf(FIRToken::comma)) {
    if (parseRUW(ruw))
      return failure();
  }

  if (parseOptionalInfo()) {
    return failure();
  }

  locationProcessor.setLoc(startTok.getLoc());

  // Transform the parsed vector type into a memory type.
  auto vectorType = type_dyn_cast<FVectorType>(type);
  if (!vectorType)
    return emitError("smem requires vector type");

  auto annotations = getConstants().emptyArrayAttr;
  StringAttr sym = {};
  auto result = SeqMemOp::create(
      builder, vectorType.getElementType(), vectorType.getNumElements(), ruw,
      id, NameKindEnum::InterestingName, annotations, sym);
  return moduleContext.addSymbolEntry(id, result, startTok.getLoc());
}

/// mem ::= 'mem' id ':' info? INDENT memField* DEDENT
/// memField ::= 'data-type' '=>' type NEWLINE
///          ::= 'depth' '=>' intLit NEWLINE
///          ::= 'read-latency' '=>' intLit NEWLINE
///          ::= 'write-latency' '=>' intLit NEWLINE
///          ::= 'read-under-write' '=>' ruw NEWLINE
///          ::= 'reader' '=>' id+ NEWLINE
///          ::= 'writer' '=>' id+ NEWLINE
///          ::= 'readwriter' '=>' id+ NEWLINE
ParseResult FIRStmtParser::parseMem(unsigned memIndent) {
  auto startTok = consumeToken(FIRToken::kw_mem);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  StringRef id;
  if (parseId(id, "expected mem name") ||
      parseToken(FIRToken::colon, "expected ':' in mem") || parseOptionalInfo())
    return failure();

  FIRRTLType type;
  int64_t depth = -1, readLatency = -1, writeLatency = -1;
  RUWBehavior ruw = RUWBehavior::Undefined;

  SmallVector<std::pair<StringAttr, Type>, 4> ports;

  // Parse all the memfield records, which are indented more than the mem.
  while (1) {
    auto nextIndent = getIndentation();
    if (!nextIndent || *nextIndent <= memIndent)
      break;

    auto spelling = getTokenSpelling();
    if (parseToken(FIRToken::identifier, "unexpected token in 'mem'") ||
        parseToken(FIRToken::equal_greater, "expected '=>' in 'mem'"))
      return failure();

    if (spelling == "data-type") {
      if (type)
        return emitError("'mem' type specified multiple times"), failure();

      if (parseType(type, "expected type in data-type declaration"))
        return failure();
      continue;
    }
    if (spelling == "depth") {
      if (parseIntLit(depth, "expected integer in depth specification"))
        return failure();
      continue;
    }
    if (spelling == "read-latency") {
      if (parseIntLit(readLatency, "expected integer latency"))
        return failure();
      continue;
    }
    if (spelling == "write-latency") {
      if (parseIntLit(writeLatency, "expected integer latency"))
        return failure();
      continue;
    }
    if (spelling == "read-under-write") {
      if (getToken().isNot(FIRToken::kw_old, FIRToken::kw_new,
                           FIRToken::kw_undefined))
        return emitError("expected specifier"), failure();

      if (parseOptionalRUW(ruw))
        return failure();
      continue;
    }

    MemOp::PortKind portKind;
    if (spelling == "reader")
      portKind = MemOp::PortKind::Read;
    else if (spelling == "writer")
      portKind = MemOp::PortKind::Write;
    else if (spelling == "readwriter")
      portKind = MemOp::PortKind::ReadWrite;
    else
      return emitError("unexpected field in 'mem' declaration"), failure();

    StringRef portName;
    if (parseId(portName, "expected port name"))
      return failure();
    auto baseType = type_dyn_cast<FIRRTLBaseType>(type);
    if (!baseType)
      return emitError("unexpected type, must be base type");
    ports.push_back({builder.getStringAttr(portName),
                     MemOp::getTypeForPort(depth, baseType, portKind)});

    while (!getIndentation().has_value()) {
      if (parseId(portName, "expected port name"))
        return failure();
      ports.push_back({builder.getStringAttr(portName),
                       MemOp::getTypeForPort(depth, baseType, portKind)});
    }
  }

  // The FIRRTL dialect requires mems to have at least one port.  Since portless
  // mems can never be referenced, it is always safe to drop them.
  if (ports.empty())
    return success();

  // Canonicalize the ports into alphabetical order.
  // TODO: Move this into MemOp construction/canonicalization.
  llvm::array_pod_sort(ports.begin(), ports.end(),
                       [](const std::pair<StringAttr, Type> *lhs,
                          const std::pair<StringAttr, Type> *rhs) -> int {
                         return lhs->first.getValue().compare(
                             rhs->first.getValue());
                       });

  auto annotations = getConstants().emptyArrayAttr;
  SmallVector<Attribute, 4> resultNames;
  SmallVector<Type, 4> resultTypes;
  SmallVector<Attribute, 4> resultAnnotations;
  for (auto p : ports) {
    resultNames.push_back(p.first);
    resultTypes.push_back(p.second);
    resultAnnotations.push_back(annotations);
  }

  locationProcessor.setLoc(startTok.getLoc());

  auto result = MemOp::create(
      builder, resultTypes, readLatency, writeLatency, depth, ruw,
      builder.getArrayAttr(resultNames), id, NameKindEnum::InterestingName,
      annotations, builder.getArrayAttr(resultAnnotations), hw::InnerSymAttr(),
      MemoryInitAttr(), StringAttr());

  UnbundledValueEntry unbundledValueEntry;
  unbundledValueEntry.reserve(result.getNumResults());
  for (size_t i = 0, e = result.getNumResults(); i != e; ++i)
    unbundledValueEntry.push_back({resultNames[i], result.getResult(i)});

  moduleContext.unbundledValues.push_back(std::move(unbundledValueEntry));
  auto entryID = UnbundledID(moduleContext.unbundledValues.size());
  return moduleContext.addSymbolEntry(id, entryID, startTok.getLoc());
}

/// node ::= 'node' id '=' exp info?
ParseResult FIRStmtParser::parseNode() {
  auto startTok = consumeToken(FIRToken::kw_node);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  StringRef id;
  Value initializer;
  if (parseId(id, "expected node name") ||
      parseToken(FIRToken::equal, "expected '=' in node") ||
      parseExp(initializer, "expected expression for node") ||
      parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  // Error out in the following conditions:
  //
  //   1. Node type is Analog (at the top level)
  //   2. Node type is not passive under an optional outer flip
  //      (analog field is okay)
  //
  // Note: (1) is more restictive than normal NodeOp verification, but
  // this is added to align with the SFC. (2) is less restrictive than
  // the SFC to accomodate for situations where the node is something
  // weird like a module output or an instance input.
  auto initializerType = type_cast<FIRRTLType>(initializer.getType());
  auto initializerBaseType =
      type_dyn_cast<FIRRTLBaseType>(initializer.getType());
  if (type_isa<AnalogType>(initializerType) ||
      !(initializerBaseType && initializerBaseType.isPassive())) {
    emitError(startTok.getLoc())
        << "Node cannot be analog and must be passive or passive under a flip "
        << initializer.getType();
    return failure();
  }

  auto annotations = getConstants().emptyArrayAttr;
  StringAttr sym = {};

  auto result = NodeOp::create(builder, initializer, id,
                               NameKindEnum::InterestingName, annotations, sym);
  return moduleContext.addSymbolEntry(id, result.getResult(),
                                      startTok.getLoc());
}

/// wire ::= 'wire' id ':' type info?
ParseResult FIRStmtParser::parseWire() {
  auto startTok = consumeToken(FIRToken::kw_wire);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  StringRef id;
  FIRRTLType type;
  if (parseId(id, "expected wire name") ||
      parseToken(FIRToken::colon, "expected ':' in wire") ||
      parseType(type, "expected wire type") || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  auto annotations = getConstants().emptyArrayAttr;
  StringAttr sym = {};

  // Names of only-nonHW should be droppable.
  auto namekind = isa<PropertyType, RefType>(type)
                      ? NameKindEnum::DroppableName
                      : NameKindEnum::InterestingName;

  auto result = WireOp::create(builder, type, id, namekind, annotations, sym);
  return moduleContext.addSymbolEntry(id, result.getResult(),
                                      startTok.getLoc());
}

/// register    ::= 'reg' id ':' type exp ('with' ':' reset_block)? info?
///
/// reset_block ::= INDENT simple_reset info? NEWLINE DEDENT
///             ::= '(' simple_reset ')'
///
/// simple_reset ::= simple_reset0
///              ::= '(' simple_reset0 ')'
///
/// simple_reset0:  'reset' '=>' '(' exp exp ')'
///
ParseResult FIRStmtParser::parseRegister(unsigned regIndent) {
  auto startTok = consumeToken(FIRToken::kw_reg);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  StringRef id;
  FIRRTLType type;
  Value clock;

  // TODO(firrtl spec): info? should come after the clock expression before
  // the 'with'.
  if (parseId(id, "expected reg name") ||
      parseToken(FIRToken::colon, "expected ':' in reg") ||
      parseType(type, "expected reg type") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(clock, "expected expression for register clock"))
    return failure();

  if (!type_isa<FIRRTLBaseType>(type))
    return emitError(startTok.getLoc(), "register must have base type");

  // Parse the 'with' specifier if present.
  Value resetSignal, resetValue;
  if (consumeIf(FIRToken::kw_with)) {
    if (removedFeature({3, 0, 0}, "'reg with' registers"))
      return failure();

    if (parseToken(FIRToken::colon, "expected ':' in reg"))
      return failure();

    // TODO(firrtl spec): Simplify the grammar for register reset logic.
    // Why allow multiple ambiguous parentheses?  Why rely on indentation at
    // all?

    // This implements what the examples have in practice.
    bool hasExtraLParen = consumeIf(FIRToken::l_paren);

    auto indent = getIndentation();
    if (!indent || *indent <= regIndent)
      if (!hasExtraLParen)
        return emitError("expected indented reset specifier in reg"), failure();

    if (parseToken(FIRToken::kw_reset, "expected 'reset' in reg") ||
        parseToken(FIRToken::equal_greater, "expected => in reset specifier") ||
        parseToken(FIRToken::l_paren, "expected '(' in reset specifier") ||
        parseExp(resetSignal, "expected expression for reset signal") ||
        parseToken(FIRToken::comma, "expected ','"))
      return failure();

    // The Scala implementation of FIRRTL represents registers without resets
    // as a self referential register... and the pretty printer doesn't print
    // the right form. Recognize that this is happening and treat it as a
    // register without a reset for compatibility.
    // TODO(firrtl scala impl): pretty print registers without resets right.
    if (getTokenSpelling() == id) {
      consumeToken();
      if (parseToken(FIRToken::r_paren, "expected ')' in reset specifier"))
        return failure();
      resetSignal = Value();
    } else {
      if (parseExp(resetValue, "expected expression for reset value") ||
          parseToken(FIRToken::r_paren, "expected ')' in reset specifier"))
        return failure();
    }

    if (hasExtraLParen &&
        parseToken(FIRToken::r_paren, "expected ')' in reset specifier"))
      return failure();
  }

  // Finally, handle the last info if present, providing location info for the
  // clock expression.
  if (parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  ArrayAttr annotations = getConstants().emptyArrayAttr;
  Value result;
  StringAttr sym = {};
  if (resetSignal)
    result =
        RegResetOp::create(builder, type, clock, resetSignal, resetValue, id,
                           NameKindEnum::InterestingName, annotations, sym)
            .getResult();
  else
    result = RegOp::create(builder, type, clock, id,
                           NameKindEnum::InterestingName, annotations, sym)
                 .getResult();
  return moduleContext.addSymbolEntry(id, result, startTok.getLoc());
}

/// registerWithReset ::= 'regreset' id ':' type exp exp exp
///
/// This syntax is only supported in FIRRTL versions >= 3.0.0.  Because this
/// syntax is only valid for >= 3.0.0, there is no need to check if the leading
/// "regreset" is part of an expression with a leading keyword.
ParseResult FIRStmtParser::parseRegisterWithReset() {
  auto startTok = consumeToken(FIRToken::kw_regreset);

  StringRef id;
  FIRRTLType type;
  Value clock, resetSignal, resetValue;

  if (parseId(id, "expected reg name") ||
      parseToken(FIRToken::colon, "expected ':' in reg") ||
      parseType(type, "expected reg type") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(clock, "expected expression for register clock") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(resetSignal, "expected expression for register reset") ||
      parseToken(FIRToken::comma, "expected ','") ||
      parseExp(resetValue, "expected expression for register reset value") ||
      parseOptionalInfo())
    return failure();

  if (!type_isa<FIRRTLBaseType>(type))
    return emitError(startTok.getLoc(), "register must have base type");

  locationProcessor.setLoc(startTok.getLoc());

  auto result =
      RegResetOp::create(builder, type, clock, resetSignal, resetValue, id,
                         NameKindEnum::InterestingName,
                         getConstants().emptyArrayAttr, StringAttr{})
          .getResult();

  return moduleContext.addSymbolEntry(id, result, startTok.getLoc());
}

/// contract ::= 'contract' (id,+ '=' exp,+) ':' info? contract_body
/// contract_body ::= simple_stmt | INDENT simple_stmt+ DEDENT
ParseResult FIRStmtParser::parseContract(unsigned blockIndent) {
  if (requireFeature(missingSpecFIRVersion, "contracts"))
    return failure();

  auto startTok = consumeToken(FIRToken::kw_contract);

  // Parse the contract results and expressions.
  SmallVector<StringRef> ids;
  SmallVector<SMLoc> locs;
  SmallVector<Value> values;
  SmallVector<Type> types;
  if (!consumeIf(FIRToken::colon)) {
    auto parseContractId = [&] {
      StringRef id;
      locs.push_back(getToken().getLoc());
      if (parseId(id, "expected contract result name"))
        return failure();
      ids.push_back(id);
      return success();
    };
    auto parseContractValue = [&] {
      Value value;
      if (parseExp(value, "expected expression for contract result"))
        return failure();
      values.push_back(value);
      types.push_back(value.getType());
      return success();
    };
    if (parseListUntil(FIRToken::equal, parseContractId) ||
        parseListUntil(FIRToken::colon, parseContractValue))
      return failure();
  }
  if (parseOptionalInfo())
    return failure();

  // Each result must have a corresponding expression assigned.
  if (ids.size() != values.size())
    return emitError(startTok.getLoc())
           << "contract requires same number of results and expressions; got "
           << ids.size() << " results and " << values.size()
           << " expressions instead";

  locationProcessor.setLoc(startTok.getLoc());

  // Add block arguments for each result and declare their names in a subscope
  // for the contract body.
  auto contract = ContractOp::create(builder, types, values);
  auto &block = contract.getBody().emplaceBlock();

  // Parse the contract body.
  {
    FIRModuleContext::ContextScope scope(moduleContext, &block);
    for (auto [id, loc, type] : llvm::zip(ids, locs, types)) {
      auto arg = block.addArgument(type, LocWithInfo(loc, this).getLoc());
      if (failed(moduleContext.addSymbolEntry(id, arg, loc)))
        return failure();
    }
    if (getIndentation() > blockIndent)
      if (parseSubBlock(block, blockIndent, SymbolRefAttr{}))
        return failure();
  }

  // Declare the results.
  for (auto [id, loc, value, result] :
       llvm::zip(ids, locs, values, contract.getResults())) {
    // Remove previous symbol to avoid duplicates
    moduleContext.removeSymbolEntry(id);
    if (failed(moduleContext.addSymbolEntry(id, result, loc)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// FIRCircuitParser
//===----------------------------------------------------------------------===//

namespace {
/// This class implements the outer level of the parser, including things
/// like circuit and module.
struct FIRCircuitParser : public FIRParser {
  explicit FIRCircuitParser(SharedParserConstants &state, FIRLexer &lexer,
                            ModuleOp mlirModule, FIRVersion version)
      : FIRParser(state, lexer, version), mlirModule(mlirModule) {}

  ParseResult
  parseCircuit(SmallVectorImpl<const llvm::MemoryBuffer *> &annotationsBuf,
               mlir::TimingScope &ts);

private:
  /// Extract Annotations from a JSON-encoded Annotation array string and add
  /// them to a vector of attributes.
  ParseResult importAnnotationsRaw(SMLoc loc, StringRef annotationsStr,
                                   SmallVectorImpl<Attribute> &attrs);

  ParseResult parseToplevelDefinition(CircuitOp circuit, unsigned indent);

  ParseResult parseClass(CircuitOp circuit, unsigned indent);
  ParseResult parseDomain(CircuitOp circuit, unsigned indent);
  ParseResult parseExtClass(CircuitOp circuit, unsigned indent);
  ParseResult parseExtModule(CircuitOp circuit, unsigned indent);
  ParseResult parseIntModule(CircuitOp circuit, unsigned indent);
  ParseResult parseModule(CircuitOp circuit, bool isPublic, unsigned indent);
  ParseResult parseFormal(CircuitOp circuit, unsigned indent);
  ParseResult parseSimulation(CircuitOp circuit, unsigned indent);
  template <class Op>
  ParseResult parseFormalLike(CircuitOp circuit, unsigned indent);

  ParseResult parseLayerName(SymbolRefAttr &result);
  ParseResult parseLayerList(SmallVectorImpl<Attribute> &result);
  ParseResult parseEnableLayerSpec(SmallVectorImpl<Attribute> &result);
  ParseResult parseKnownLayerSpec(SmallVectorImpl<Attribute> &result);
  ParseResult parseModuleLayerSpec(ArrayAttr &enabledLayers);
  ParseResult parseExtModuleLayerSpec(ArrayAttr &enabledLayers,
                                      ArrayAttr &knownLayers);

  ParseResult parsePortList(SmallVectorImpl<PortInfo> &resultPorts,
                            SmallVectorImpl<SMLoc> &resultPortLocs,
                            unsigned indent);
  ParseResult parseParameterList(ArrayAttr &resultParameters);
  ParseResult parseRefList(ArrayRef<PortInfo> portList,
                           ArrayAttr &internalPathsResult);

  ParseResult skipToModuleEnd(unsigned indent);

  ParseResult parseTypeDecl();

  ParseResult parseOptionDecl(CircuitOp circuit);

  ParseResult parseLayer(CircuitOp circuit);

  struct DeferredModuleToParse {
    FModuleLike moduleOp;
    SmallVector<SMLoc> portLocs;
    FIRLexerCursor lexerCursor;
    unsigned indent;
  };

  ParseResult parseModuleBody(const SymbolTable &circuitSymTbl,
                              DeferredModuleToParse &deferredModule,
                              InnerSymFixups &fixups);

  SmallVector<DeferredModuleToParse, 0> deferredModules;

  SmallVector<InnerSymFixups, 0> moduleFixups;

  hw::InnerSymbolNamespaceCollection innerSymbolNamespaces;

  ModuleOp mlirModule;
};

} // end anonymous namespace
ParseResult
FIRCircuitParser::importAnnotationsRaw(SMLoc loc, StringRef annotationsStr,
                                       SmallVectorImpl<Attribute> &attrs) {

  auto annotations = json::parse(annotationsStr);
  if (auto err = annotations.takeError()) {
    handleAllErrors(std::move(err), [&](const json::ParseError &a) {
      auto diag = emitError(loc, "Failed to parse JSON Annotations");
      diag.attachNote() << a.message();
    });
    return failure();
  }

  json::Path::Root root;
  llvm::StringMap<ArrayAttr> thisAnnotationMap;
  if (!importAnnotationsFromJSONRaw(annotations.get(), attrs, root,
                                    getContext())) {
    auto diag = emitError(loc, "Invalid/unsupported annotation format");
    std::string jsonErrorMessage =
        "See inline comments for problem area in JSON:\n";
    llvm::raw_string_ostream s(jsonErrorMessage);
    root.printErrorContext(annotations.get(), s);
    diag.attachNote() << jsonErrorMessage;
    return failure();
  }

  return success();
}

ParseResult FIRCircuitParser::parseLayerName(SymbolRefAttr &result) {
  auto *context = getContext();
  SmallVector<StringRef> strings;
  do {
    StringRef name;
    if (parseId(name, "expected layer name"))
      return failure();
    strings.push_back(name);
  } while (consumeIf(FIRToken::period));

  SmallVector<FlatSymbolRefAttr> nested;
  nested.reserve(strings.size() - 1);
  for (unsigned i = 1, e = strings.size(); i < e; ++i)
    nested.push_back(FlatSymbolRefAttr::get(context, strings[i]));

  result = SymbolRefAttr::get(context, strings[0], nested);
  return success();
}

ParseResult FIRCircuitParser::parseModuleLayerSpec(ArrayAttr &enabledLayers) {
  SmallVector<Attribute> enabledLayersBuffer;
  while (true) {
    auto tokenKind = getToken().getKind();
    // Parse an enablelayer spec.
    if (tokenKind == FIRToken::kw_enablelayer) {
      if (parseEnableLayerSpec(enabledLayersBuffer))
        return failure();
      continue;
    }
    // Didn't parse a layer spec.
    break;
  }

  if (enabledLayersBuffer.size() != 0)
    if (requireFeature({4, 0, 0}, "modules with layers enabled"))
      return failure();

  enabledLayers = ArrayAttr::get(getContext(), enabledLayersBuffer);
  return success();
}

ParseResult FIRCircuitParser::parseExtModuleLayerSpec(ArrayAttr &enabledLayers,
                                                      ArrayAttr &knownLayers) {
  SmallVector<Attribute> enabledLayersBuffer;
  SmallVector<Attribute> knownLayersBuffer;
  while (true) {
    auto tokenKind = getToken().getKind();
    // Parse an enablelayer spec.
    if (tokenKind == FIRToken::kw_enablelayer) {
      if (parseEnableLayerSpec(enabledLayersBuffer))
        return failure();
      continue;
    }
    // Parse a knownlayer spec.
    if (tokenKind == FIRToken::kw_knownlayer) {
      if (parseKnownLayerSpec(knownLayersBuffer))
        return failure();
      continue;
    }
    // Didn't parse a layer spec.
    break;
  }

  if (enabledLayersBuffer.size() != 0)
    if (requireFeature({4, 0, 0}, "extmodules with layers enabled"))
      return failure();

  if (knownLayersBuffer.size() != 0)
    if (requireFeature(nextFIRVersion, "extmodules with known layers"))
      return failure();

  enabledLayers = ArrayAttr::get(getContext(), enabledLayersBuffer);
  knownLayers = ArrayAttr::get(getContext(), knownLayersBuffer);
  return success();
}

ParseResult
FIRCircuitParser::parseLayerList(SmallVectorImpl<Attribute> &result) {
  do {
    SymbolRefAttr layer;
    if (parseLayerName(layer))
      return failure();
    result.push_back(layer);
  } while (consumeIf(FIRToken::comma));
  return success();
}

ParseResult
FIRCircuitParser::parseEnableLayerSpec(SmallVectorImpl<Attribute> &result) {
  consumeToken(FIRToken::kw_enablelayer);
  return parseLayerList(result);
}

ParseResult
FIRCircuitParser::parseKnownLayerSpec(SmallVectorImpl<Attribute> &result) {
  consumeToken(FIRToken::kw_knownlayer);
  return parseLayerList(result);
}

/// portlist ::= port*
/// port     ::= dir id ':' type info? NEWLINE
/// dir      ::= 'input' | 'output'
ParseResult
FIRCircuitParser::parsePortList(SmallVectorImpl<PortInfo> &resultPorts,
                                SmallVectorImpl<SMLoc> &resultPortLocs,
                                unsigned indent) {
  // Parse any ports.
  while (getToken().isAny(FIRToken::kw_input, FIRToken::kw_output) &&
         // Must be nested under the module.
         getIndentation() > indent) {

    // We need one token lookahead to resolve the ambiguity between:
    // output foo             ; port
    // output <= input        ; identifier expression
    // output.thing <= input  ; identifier expression
    auto backtrackState = getLexer().getCursor();

    bool isOutput = getToken().is(FIRToken::kw_output);
    consumeToken();

    // If we have something that isn't a keyword then this must be an
    // identifier, not an input/output marker.
    if (!getToken().isAny(FIRToken::identifier, FIRToken::literal_identifier) &&
        !getToken().isKeyword()) {
      backtrackState.restore(getLexer());
      break;
    }

    StringAttr name;
    FIRRTLType type;
    LocWithInfo info(getToken().getLoc(), this);
    if (parseId(name, "expected port name") ||
        parseToken(FIRToken::colon, "expected ':' in port definition") ||
        parseType(type, "expected a type in port declaration") ||
        info.parseOptionalInfo())
      return failure();

    StringAttr innerSym = {};
    resultPorts.push_back(
        {name, type, direction::get(isOutput), innerSym, info.getLoc()});
    resultPortLocs.push_back(info.getFIRLoc());
  }

  // Check for port name collisions.
  SmallDenseMap<Attribute, SMLoc> portIds;
  for (auto portAndLoc : llvm::zip(resultPorts, resultPortLocs)) {
    PortInfo &port = std::get<0>(portAndLoc);
    auto &entry = portIds[port.name];
    if (!entry.isValid()) {
      entry = std::get<1>(portAndLoc);
      continue;
    }

    emitError(std::get<1>(portAndLoc),
              "redefinition of name '" + port.getName() + "'")
            .attachNote(translateLocation(entry))
        << "previous definition here";
    return failure();
  }

  return success();
}

/// ref-list ::= ref*
/// ref ::= 'ref' static_reference 'is' StringLit NEWLIN
ParseResult FIRCircuitParser::parseRefList(ArrayRef<PortInfo> portList,
                                           ArrayAttr &internalPathsResult) {
  struct RefStatementInfo {
    StringAttr refName;
    InternalPathAttr resolvedPath;
    SMLoc loc;
  };

  SmallVector<RefStatementInfo> refStatements;
  SmallPtrSet<StringAttr, 8> seenNames;
  SmallPtrSet<StringAttr, 8> seenRefs;

  // Ref statements were added in 2.0.0 and removed in 4.0.0.
  if (getToken().is(FIRToken::kw_ref) &&
      (requireFeature({2, 0, 0}, "ref statements") ||
       removedFeature({4, 0, 0}, "ref statements")))
    return failure();

  // Parse the ref statements.
  while (consumeIf(FIRToken::kw_ref)) {
    auto loc = getToken().getLoc();
    // ref x is "a.b.c"
    // Support "ref x.y is " once aggregate-of-ref supported.
    StringAttr refName;
    if (parseId(refName, "expected ref name"))
      return failure();
    if (consumeIf(FIRToken::period) || consumeIf(FIRToken::l_square))
      return emitError(
          loc, "ref statements for aggregate elements not yet supported");
    if (parseToken(FIRToken::kw_is, "expected 'is' in ref statement"))
      return failure();

    if (!seenRefs.insert(refName).second)
      return emitError(loc, "duplicate ref statement for '" + refName.strref() +
                                "'");

    auto kind = getToken().getKind();
    if (kind != FIRToken::string)
      return emitError(loc, "expected string in ref statement");
    auto resolved = InternalPathAttr::get(
        getContext(),
        StringAttr::get(getContext(), getToken().getStringValue()));
    consumeToken(FIRToken::string);

    refStatements.push_back(RefStatementInfo{refName, resolved, loc});
  }

  // Build paths array.  One entry for each ref-type port, empty for others.
  SmallVector<Attribute> internalPaths(portList.size(),
                                       InternalPathAttr::get(getContext()));

  llvm::SmallBitVector usedRefs(refStatements.size());
  size_t matchedPaths = 0;
  for (auto [idx, port] : llvm::enumerate(portList)) {
    if (!type_isa<RefType>(port.type))
      continue;

    // Reject input reftype ports on extmodule's per spec,
    // as well as on intmodule's which is not mentioned in spec.
    if (!port.isOutput())
      return mlir::emitError(
          port.loc,
          "references in ports must be output on extmodule and intmodule");
    auto *refStmtIt =
        llvm::find_if(refStatements, [pname = port.name](const auto &r) {
          return r.refName == pname;
        });
    // Error if ref statements are present but none found for this port.
    if (refStmtIt == refStatements.end()) {
      if (!refStatements.empty())
        return mlir::emitError(port.loc, "no ref statement found for ref port ")
            .append(port.name);
      continue;
    }

    usedRefs.set(std::distance(refStatements.begin(), refStmtIt));
    internalPaths[idx] = refStmtIt->resolvedPath;
    ++matchedPaths;
  }

  if (!refStatements.empty() && matchedPaths != refStatements.size()) {
    assert(matchedPaths < refStatements.size());
    assert(!usedRefs.all());
    auto idx = usedRefs.find_first_unset();
    assert(idx != -1);
    return emitError(refStatements[idx].loc, "unused ref statement");
  }

  if (matchedPaths)
    internalPathsResult = ArrayAttr::get(getContext(), internalPaths);
  return success();
}

/// We're going to defer parsing this module, so just skip tokens until we
/// get to the next module or the end of the file.
ParseResult FIRCircuitParser::skipToModuleEnd(unsigned indent) {
  while (true) {
    switch (getToken().getKind()) {

    // End of file or invalid token will be handled by outer level.
    case FIRToken::eof:
    case FIRToken::error:
      return success();

    // If we got to the next top-level declaration, then we're done.
    case FIRToken::kw_class:
    case FIRToken::kw_domain:
    case FIRToken::kw_declgroup:
    case FIRToken::kw_extclass:
    case FIRToken::kw_extmodule:
    case FIRToken::kw_intmodule:
    case FIRToken::kw_formal:
    case FIRToken::kw_module:
    case FIRToken::kw_public:
    case FIRToken::kw_layer:
    case FIRToken::kw_option:
    case FIRToken::kw_simulation:
    case FIRToken::kw_type:
      // All module declarations should have the same indentation
      // level. Use this fact to differentiate between module
      // declarations and usages of "module" as identifiers.
      if (getIndentation() == indent)
        return success();
      [[fallthrough]];
    default:
      consumeToken();
      break;
    }
  }
}

/// parameter-list ::= parameter*
/// parameter ::= 'parameter' param NEWLINE
ParseResult FIRCircuitParser::parseParameterList(ArrayAttr &resultParameters) {
  SmallVector<Attribute, 8> parameters;
  SmallPtrSet<StringAttr, 8> seen;
  while (consumeIf(FIRToken::kw_parameter)) {
    StringAttr name;
    Attribute value;
    SMLoc loc;
    if (parseParameter(name, value, loc))
      return failure();
    auto typedValue = dyn_cast<TypedAttr>(value);
    if (!typedValue)
      return emitError(loc)
             << "invalid value for parameter '" << name.getValue() << "'";
    if (!seen.insert(name).second)
      return emitError(loc,
                       "redefinition of parameter '" + name.getValue() + "'");
    parameters.push_back(ParamDeclAttr::get(name, typedValue));
  }
  resultParameters = ArrayAttr::get(getContext(), parameters);
  return success();
}

/// class ::= 'class' id ':' info? INDENT portlist simple_stmt_block DEDENT
ParseResult FIRCircuitParser::parseClass(CircuitOp circuit, unsigned indent) {
  StringAttr name;
  SmallVector<PortInfo, 8> portList;
  SmallVector<SMLoc> portLocs;
  LocWithInfo info(getToken().getLoc(), this);

  if (requireFeature(missingSpecFIRVersion, "classes"))
    return failure();

  consumeToken(FIRToken::kw_class);
  if (parseId(name, "expected class name") ||
      parseToken(FIRToken::colon, "expected ':' in class definition") ||
      info.parseOptionalInfo() || parsePortList(portList, portLocs, indent))
    return failure();

  if (name == circuit.getName())
    return mlir::emitError(info.getLoc(),
                           "class cannot be the top of a circuit");

  for (auto &portInfo : portList)
    if (!isa<PropertyType>(portInfo.type))
      return mlir::emitError(portInfo.loc,
                             "ports on classes must be properties");

  // build it
  auto builder = circuit.getBodyBuilder();
  auto classOp = ClassOp::create(builder, info.getLoc(), name, portList);
  classOp.setPrivate();
  deferredModules.emplace_back(
      DeferredModuleToParse{classOp, portLocs, getLexer().getCursor(), indent});

  // Stash the class name -> op in the constants, so we can resolve Inst types.
  getConstants().classMap[name.getValue()] = classOp;
  return skipToModuleEnd(indent);
}

/// domain ::= 'domain' id ':' info?
ParseResult FIRCircuitParser::parseDomain(CircuitOp circuit, unsigned indent) {
  consumeToken(FIRToken::kw_domain);

  StringAttr name;
  LocWithInfo info(getToken().getLoc(), this);
  if (parseId(name, "domain name") ||
      parseToken(FIRToken::colon, "expected ':' after domain definition") ||
      info.parseOptionalInfo())
    return failure();

  auto builder = circuit.getBodyBuilder();
  DomainOp::create(builder, info.getLoc(), name)
      ->getRegion(0)
      .push_back(new Block());

  return success();
}

/// extclass ::= 'extclass' id ':' info? INDENT portlist DEDENT
ParseResult FIRCircuitParser::parseExtClass(CircuitOp circuit,
                                            unsigned indent) {
  StringAttr name;
  SmallVector<PortInfo, 8> portList;
  SmallVector<SMLoc> portLocs;
  LocWithInfo info(getToken().getLoc(), this);

  if (requireFeature(missingSpecFIRVersion, "classes"))
    return failure();

  consumeToken(FIRToken::kw_extclass);
  if (parseId(name, "expected extclass name") ||
      parseToken(FIRToken::colon, "expected ':' in extclass definition") ||
      info.parseOptionalInfo() || parsePortList(portList, portLocs, indent))
    return failure();

  if (name == circuit.getName())
    return mlir::emitError(info.getLoc(),
                           "extclass cannot be the top of a circuit");

  for (auto &portInfo : portList)
    if (!isa<PropertyType>(portInfo.type))
      return mlir::emitError(portInfo.loc,
                             "ports on extclasses must be properties");

  // Build it
  auto builder = circuit.getBodyBuilder();
  auto extClassOp = ExtClassOp::create(builder, info.getLoc(), name, portList);

  // Stash the class name -> op in the constants, so we can resolve Inst types.
  getConstants().classMap[name.getValue()] = extClassOp;
  return skipToModuleEnd(indent);
}

/// extmodule ::=
///        'extmodule' id ':' info?
///        INDENT portlist defname? parameter-list ref-list DEDENT
/// defname   ::= 'defname' '=' id NEWLINE
ParseResult FIRCircuitParser::parseExtModule(CircuitOp circuit,
                                             unsigned indent) {
  StringAttr name;
  ArrayAttr enabledLayers;
  ArrayAttr knownLayers;
  SmallVector<PortInfo, 8> portList;
  SmallVector<SMLoc> portLocs;
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_extmodule);
  if (parseId(name, "expected extmodule name") ||
      parseExtModuleLayerSpec(enabledLayers, knownLayers) ||
      parseToken(FIRToken::colon, "expected ':' in extmodule definition") ||
      info.parseOptionalInfo() || parsePortList(portList, portLocs, indent))
    return failure();

  StringRef defName;
  if (consumeIf(FIRToken::kw_defname)) {
    if (parseToken(FIRToken::equal, "expected '=' in defname") ||
        parseId(defName, "expected defname name"))
      return failure();
  }

  ArrayAttr parameters;
  ArrayAttr internalPaths;
  if (parseParameterList(parameters) || parseRefList(portList, internalPaths))
    return failure();

  if (version >= FIRVersion({4, 0, 0})) {
    for (auto [pi, loc] : llvm::zip_equal(portList, portLocs)) {
      if (auto ftype = type_dyn_cast<FIRRTLType>(pi.type)) {
        if (ftype.hasUninferredWidth())
          return emitError(loc, "extmodule port must have known width");
      }
    }
  }

  auto builder = circuit.getBodyBuilder();
  auto isMainModule = (name == circuit.getName());
  auto convention =
      (isMainModule && getConstants().options.scalarizePublicModules) ||
              getConstants().options.scalarizeExtModules
          ? Convention::Scalarized
          : Convention::Internal;
  auto conventionAttr = ConventionAttr::get(getContext(), convention);
  auto annotations = ArrayAttr::get(getContext(), {});
  auto extModuleOp = FExtModuleOp::create(
      builder, info.getLoc(), name, conventionAttr, portList, knownLayers,
      defName, annotations, parameters, internalPaths, enabledLayers);
  auto visibility = isMainModule ? SymbolTable::Visibility::Public
                                 : SymbolTable::Visibility::Private;
  SymbolTable::setSymbolVisibility(extModuleOp, visibility);
  return success();
}

/// intmodule ::=
///        'intmodule' id ':' info?
///        INDENT portlist intname parameter-list ref-list DEDENT
/// intname   ::= 'intrinsic' '=' id NEWLINE
ParseResult FIRCircuitParser::parseIntModule(CircuitOp circuit,
                                             unsigned indent) {
  StringAttr name;
  StringRef intName;
  ArrayAttr enabledLayers;
  SmallVector<PortInfo, 8> portList;
  SmallVector<SMLoc> portLocs;
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_intmodule);
  if (parseId(name, "expected intmodule name") ||
      parseModuleLayerSpec(enabledLayers) ||
      parseToken(FIRToken::colon, "expected ':' in intmodule definition") ||
      info.parseOptionalInfo() || parsePortList(portList, portLocs, indent) ||
      parseToken(FIRToken::kw_intrinsic, "expected 'intrinsic'") ||
      parseToken(FIRToken::equal, "expected '=' in intrinsic") ||
      parseId(intName, "expected intrinsic name"))
    return failure();

  ArrayAttr parameters;
  ArrayAttr internalPaths;
  if (parseParameterList(parameters) || parseRefList(portList, internalPaths))
    return failure();

  ArrayAttr annotations = getConstants().emptyArrayAttr;
  auto builder = circuit.getBodyBuilder();
  FIntModuleOp::create(builder, info.getLoc(), name, portList, intName,
                       annotations, parameters, internalPaths, enabledLayers)
      .setPrivate();
  return success();
}

/// module ::= 'module' id ':' info? INDENT portlist simple_stmt_block DEDENT
ParseResult FIRCircuitParser::parseModule(CircuitOp circuit, bool isPublic,
                                          unsigned indent) {
  StringAttr name;
  SmallVector<PortInfo, 8> portList;
  SmallVector<SMLoc> portLocs;
  ArrayAttr enabledLayers;
  auto modLoc = getToken().getLoc();
  LocWithInfo info(modLoc, this);
  consumeToken(FIRToken::kw_module);
  if (parseId(name, "expected module name") ||
      parseModuleLayerSpec(enabledLayers) ||
      parseToken(FIRToken::colon, "expected ':' in module definition") ||
      info.parseOptionalInfo() || parsePortList(portList, portLocs, indent))
    return failure();

  // The main module is implicitly public.
  if (name == circuit.getName()) {
    if (!isPublic && removedFeature({4, 0, 0}, "private main modules", modLoc))
      return failure();
    isPublic = true;
  }

  if (isPublic && version >= FIRVersion({4, 0, 0})) {
    for (auto [pi, loc] : llvm::zip_equal(portList, portLocs)) {
      if (auto ftype = type_dyn_cast<FIRRTLType>(pi.type)) {
        if (ftype.hasUninferredWidth())
          return emitError(loc, "public module port must have known width");
        if (ftype.hasUninferredReset())
          return emitError(loc,
                           "public module port must have concrete reset type");
      }
    }
  }

  ArrayAttr annotations = getConstants().emptyArrayAttr;
  auto convention = Convention::Internal;
  if (isPublic && getConstants().options.scalarizePublicModules)
    convention = Convention::Scalarized;
  if (!isPublic && getConstants().options.scalarizeInternalModules)
    convention = Convention::Scalarized;
  auto conventionAttr = ConventionAttr::get(getContext(), convention);
  auto builder = circuit.getBodyBuilder();
  auto moduleOp =
      FModuleOp::create(builder, info.getLoc(), name, conventionAttr, portList,
                        annotations, enabledLayers);

  auto visibility = isPublic ? SymbolTable::Visibility::Public
                             : SymbolTable::Visibility::Private;
  SymbolTable::setSymbolVisibility(moduleOp, visibility);

  // Parse the body of this module after all prototypes have been parsed. This
  // allows us to handle forward references correctly.
  deferredModules.emplace_back(DeferredModuleToParse{
      moduleOp, portLocs, getLexer().getCursor(), indent});

  if (skipToModuleEnd(indent))
    return failure();
  return success();
}

/// formal ::= 'formal' formal-like
ParseResult FIRCircuitParser::parseFormal(CircuitOp circuit, unsigned indent) {
  consumeToken(FIRToken::kw_formal);
  return parseFormalLike<FormalOp>(circuit, indent);
}

/// simulation ::= 'simulation' formal-like
ParseResult FIRCircuitParser::parseSimulation(CircuitOp circuit,
                                              unsigned indent) {
  consumeToken(FIRToken::kw_simulation);
  return parseFormalLike<SimulationOp>(circuit, indent);
}

/// formal-like ::= formal-like-old | formal-like-new
/// formal-like-old ::= id 'of' id ',' 'bound' '=' int info?
/// formal-like-new ::= id 'of' id ':' info? INDENT (param NEWLINE)* DEDENT
template <class Op>
ParseResult FIRCircuitParser::parseFormalLike(CircuitOp circuit,
                                              unsigned indent) {
  StringRef id, moduleName;
  int64_t bound = 0;
  LocWithInfo info(getToken().getLoc(), this);
  auto builder = circuit.getBodyBuilder();

  // Parse the name and target module of the test.
  if (parseId(id, "expected test name") ||
      parseToken(FIRToken::kw_of, "expected 'of' in test") ||
      parseId(moduleName, "expected module name"))
    return failure();

  // TODO: Remove the old `, bound = N` variant in favor of the new parameters.
  NamedAttrList params;
  if (consumeIf(FIRToken::comma)) {
    // Parse the old style declaration with a `, bound = N` suffix.
    if (getToken().isNot(FIRToken::identifier) || getTokenSpelling() != "bound")
      return emitError("expected 'bound' after ','");
    consumeToken();
    if (parseToken(FIRToken::equal, "expected '=' after 'bound'") ||
        parseIntLit(bound, "expected integer bound after '='"))
      return failure();
    if (bound <= 0)
      return emitError("bound must be a positive integer");
    if (info.parseOptionalInfo())
      return failure();
    params.set("bound", builder.getIntegerAttr(builder.getI32Type(), bound));
  } else {
    // Parse the new style declaration with a `:` and parameter list.
    if (parseToken(FIRToken::colon, "expected ':' in test") ||
        info.parseOptionalInfo())
      return failure();
    while (getIndentation() > indent) {
      StringAttr paramName;
      Attribute paramValue;
      SMLoc paramLoc;
      if (parseParameter(paramName, paramValue, paramLoc,
                         /*allowAggregates=*/true))
        return failure();
      if (params.set(paramName, paramValue))
        return emitError(paramLoc, "redefinition of parameter '" +
                                       paramName.getValue() + "'");
    }
  }

  Op::create(builder, info.getLoc(), id, moduleName,
             params.getDictionary(getContext()));
  return success();
}

ParseResult FIRCircuitParser::parseToplevelDefinition(CircuitOp circuit,
                                                      unsigned indent) {
  switch (getToken().getKind()) {
  case FIRToken::kw_class:
    return parseClass(circuit, indent);
  case FIRToken::kw_declgroup:
    if (requireFeature({3, 2, 0}, "optional groups") ||
        removedFeature({3, 3, 0}, "optional groups"))
      return failure();
    return parseLayer(circuit);
  case FIRToken::kw_domain:
    if (requireFeature(missingSpecFIRVersion, "domains"))
      return failure();
    return parseDomain(circuit, indent);
  case FIRToken::kw_extclass:
    return parseExtClass(circuit, indent);
  case FIRToken::kw_extmodule:
    return parseExtModule(circuit, indent);
  case FIRToken::kw_formal:
    if (requireFeature({4, 0, 0}, "formal tests"))
      return failure();
    return parseFormal(circuit, indent);
  case FIRToken::kw_intmodule:
    if (requireFeature({1, 2, 0}, "intrinsic modules") ||
        removedFeature({4, 0, 0}, "intrinsic modules"))
      return failure();
    return parseIntModule(circuit, indent);
  case FIRToken::kw_layer:
    if (requireFeature({3, 3, 0}, "layers"))
      return failure();
    return parseLayer(circuit);
  case FIRToken::kw_module:
    return parseModule(circuit, /*isPublic=*/false, indent);
  case FIRToken::kw_public:
    if (requireFeature({3, 3, 0}, "public modules"))
      return failure();
    consumeToken();
    if (getToken().getKind() == FIRToken::kw_module)
      return parseModule(circuit, /*isPublic=*/true, indent);
    return emitError(getToken().getLoc(), "only modules may be public");
  case FIRToken::kw_simulation:
    if (requireFeature(nextFIRVersion, "simulation tests"))
      return failure();
    return parseSimulation(circuit, indent);
  case FIRToken::kw_type:
    return parseTypeDecl();
  case FIRToken::kw_option:
    if (requireFeature(missingSpecFIRVersion, "option groups/instance choices"))
      return failure();
    return parseOptionDecl(circuit);
  default:
    return emitError(getToken().getLoc(), "unknown toplevel definition");
  }
}

// Parse a type declaration.
ParseResult FIRCircuitParser::parseTypeDecl() {
  StringRef id;
  FIRRTLType type;
  consumeToken();
  auto loc = getToken().getLoc();

  if (getToken().isKeyword())
    return emitError(loc) << "cannot use keyword '" << getToken().getSpelling()
                          << "' for type alias name";

  if (parseId(id, "expected type name") ||
      parseToken(FIRToken::equal, "expected '=' in type decl") ||
      parseType(type, "expected a type"))
    return failure();
  auto name = StringAttr::get(type.getContext(), id);
  // Create type alias only for base types. Otherwise just pass through the
  // type.
  if (auto base = type_dyn_cast<FIRRTLBaseType>(type))
    type = BaseTypeAliasType::get(name, base);
  else
    emitWarning(loc)
        << "type alias for non-base type " << type
        << " is currently not supported. Type alias is stripped immediately";

  if (!getConstants().aliasMap.insert({id, type}).second)
    return emitError(loc) << "type alias `" << name.getValue()
                          << "` is already defined";
  return success();
}

// Parse an option group declaration.
ParseResult FIRCircuitParser::parseOptionDecl(CircuitOp circuit) {
  StringRef id;
  consumeToken();
  auto loc = getToken().getLoc();

  LocWithInfo info(getToken().getLoc(), this);
  if (parseId(id, "expected an option group name") ||
      parseToken(FIRToken::colon,
                 "expected ':' after option group definition") ||
      info.parseOptionalInfo())
    return failure();

  auto builder = OpBuilder::atBlockEnd(circuit.getBodyBlock());
  auto optionOp = OptionOp::create(builder, info.getLoc(), id);
  auto *block = new Block;
  optionOp.getBody().push_back(block);
  builder.setInsertionPointToEnd(block);

  auto baseIndent = getIndentation();
  StringSet<> cases;
  while (getIndentation() == baseIndent) {
    StringRef id;
    LocWithInfo caseInfo(getToken().getLoc(), this);
    if (parseId(id, "expected an option case ID") ||
        caseInfo.parseOptionalInfo())
      return failure();

    if (!cases.insert(id).second)
      return emitError(loc)
             << "duplicate option case definition '" << id << "'";

    OptionCaseOp::create(builder, caseInfo.getLoc(), id);
  }

  return success();
}

// Parse a layer definition.
ParseResult FIRCircuitParser::parseLayer(CircuitOp circuit) {
  auto baseIndent = getIndentation();

  // A stack of all layers that are possibly parents of the current layer.
  SmallVector<std::pair<std::optional<unsigned>, LayerOp>> layerStack;

  // Parse a single layer and add it to the layerStack.
  auto parseOne = [&](Block *block) -> ParseResult {
    auto indent = getIndentation();
    StringRef id, convention;
    LocWithInfo info(getToken().getLoc(), this);
    consumeToken();
    if (parseId(id, "expected layer name") ||
        parseToken(FIRToken::comma, "expected ','") ||
        parseGetSpelling(convention))
      return failure();

    auto layerConvention = symbolizeLayerConvention(convention);
    if (!layerConvention) {
      emitError() << "unknown convention '" << convention
                  << "' (did you misspell it?)";
      return failure();
    }
    if (layerConvention == LayerConvention::Inline &&
        requireFeature({4, 1, 0}, "inline layers"))
      return failure();
    consumeToken();

    hw::OutputFileAttr outputDir;
    if (consumeIf(FIRToken::comma)) {
      if (getToken().getKind() == FIRToken::string) {
        auto text = getToken().getStringValue();
        if (text.empty())
          return emitError() << "output directory must not be blank";
        outputDir = hw::OutputFileAttr::getAsDirectory(getContext(), text);
        consumeToken(FIRToken::string);
      }
    }

    if (parseToken(FIRToken::colon, "expected ':' after layer definition") ||
        info.parseOptionalInfo())
      return failure();
    auto builder = OpBuilder::atBlockEnd(block);
    // Create the layer definition and give it an empty block.
    auto layerOp =
        LayerOp::create(builder, info.getLoc(), id, *layerConvention);
    layerOp->getRegion(0).push_back(new Block());
    if (outputDir)
      layerOp->setAttr("output_file", outputDir);
    layerStack.push_back({indent, layerOp});
    return success();
  };

  if (parseOne(circuit.getBodyBlock()))
    return failure();

  // Parse any nested layers.
  while (getIndentation() > baseIndent) {
    switch (getToken().getKind()) {
    case FIRToken::kw_declgroup:
    case FIRToken::kw_layer: {
      // Pop nested layers off the stack until we find out what layer to insert
      // this into.
      while (layerStack.back().first >= getIndentation())
        layerStack.pop_back();
      auto parentLayer = layerStack.back().second;
      if (parseOne(&parentLayer.getBody().front()))
        return failure();
      break;
    }
    default:
      return emitError("expected 'layer'"), failure();
    }
  }

  return success();
}

// Parse the body of this module.
ParseResult
FIRCircuitParser::parseModuleBody(const SymbolTable &circuitSymTbl,
                                  DeferredModuleToParse &deferredModule,
                                  InnerSymFixups &fixups) {
  FModuleLike moduleOp = deferredModule.moduleOp;
  auto &body = moduleOp->getRegion(0).front();
  auto &portLocs = deferredModule.portLocs;

  // We parse the body of this module with its own lexer, enabling parallel
  // parsing with the rest of the other module bodies.
  FIRLexer moduleBodyLexer(getLexer().getSourceMgr(), getContext());

  // Reset the parser/lexer state back to right after the port list.
  deferredModule.lexerCursor.restore(moduleBodyLexer);

  FIRModuleContext moduleContext(&body, getConstants(), moduleBodyLexer,
                                 version);

  // Install all of the ports into the symbol table, associated with their
  // block arguments.
  auto portList = moduleOp.getPorts();
  auto portArgs = body.getArguments();
  for (auto tuple : llvm::zip(portList, portLocs, portArgs)) {
    PortInfo &port = std::get<0>(tuple);
    llvm::SMLoc loc = std::get<1>(tuple);
    BlockArgument portArg = std::get<2>(tuple);
    assert(!port.sym);
    if (moduleContext.addSymbolEntry(port.getName(), portArg, loc))
      return failure();
  }

  FIRStmtParser stmtParser(body, moduleContext, fixups, circuitSymTbl, version);

  // Parse the moduleBlock.
  auto result = stmtParser.parseSimpleStmtBlock(deferredModule.indent);
  if (failed(result))
    return result;

  // Scan for printf-encoded verif's to error on their use, no longer supported.
  {
    size_t numVerifPrintfs = 0;
    std::optional<Location> printfLoc;

    deferredModule.moduleOp.walk([&](PrintFOp printFOp) {
      if (!circt::firrtl::isRecognizedPrintfEncodedVerif(printFOp))
        return;
      ++numVerifPrintfs;
      if (!printfLoc)
        printfLoc = printFOp.getLoc();
    });

    if (numVerifPrintfs > 0) {
      auto diag =
          mlir::emitError(deferredModule.moduleOp.getLoc(), "module contains ")
          << numVerifPrintfs
          << " printf-encoded verification operation(s), which are no longer "
             "supported.";
      diag.attachNote(*printfLoc)
          << "example printf here, this is now just a printf and nothing more";
      diag.attachNote() << "For more information, see "
                           "https://github.com/llvm/circt/issues/6970";
      return diag;
    }
  }

  return success();
}

/// file ::= circuit
/// versionHeader ::= 'FIRRTL' 'version' versionLit NEWLINE
/// circuit ::= versionHeader? 'circuit' id ':' info? INDENT module* DEDENT EOF
///
/// If non-null, annotationsBuf is a memory buffer containing JSON annotations.
///
ParseResult FIRCircuitParser::parseCircuit(
    SmallVectorImpl<const llvm::MemoryBuffer *> &annotationsBufs,
    mlir::TimingScope &ts) {

  auto indent = getIndentation();
  if (parseToken(FIRToken::kw_FIRRTL, "expected 'FIRRTL'"))
    return failure();
  if (!indent.has_value())
    return emitError("'FIRRTL' must be first token on its line");
  if (parseToken(FIRToken::kw_version, "expected version after 'FIRRTL'") ||
      parseVersionLit("expected version literal"))
    return failure();
  indent = getIndentation();

  if (!indent.has_value())
    return emitError("'circuit' must be first token on its line");
  unsigned circuitIndent = *indent;

  LocWithInfo info(getToken().getLoc(), this);
  StringAttr name;
  SMLoc inlineAnnotationsLoc;
  StringRef inlineAnnotations;

  // A file must contain a top level `circuit` definition.
  if (parseToken(FIRToken::kw_circuit,
                 "expected a top-level 'circuit' definition") ||
      parseId(name, "expected circuit name") ||
      parseToken(FIRToken::colon, "expected ':' in circuit definition") ||
      parseOptionalAnnotations(inlineAnnotationsLoc, inlineAnnotations) ||
      info.parseOptionalInfo())
    return failure();

  // Create the top-level circuit op in the MLIR module.
  OpBuilder b(mlirModule.getBodyRegion());
  auto circuit = CircuitOp::create(b, info.getLoc(), name);

  // A timer to get execution time of annotation parsing.
  auto parseAnnotationTimer = ts.nest("Parse annotations");

  // Deal with any inline annotations, if they exist.  These are processed
  // first to place any annotations from an annotation file *after* the inline
  // annotations.  While arbitrary, this makes the annotation file have
  // "append" semantics.
  SmallVector<Attribute> annos;
  if (!inlineAnnotations.empty())
    if (importAnnotationsRaw(inlineAnnotationsLoc, inlineAnnotations, annos))
      return failure();

  // Deal with the annotation file if one was specified
  for (auto *annotationsBuf : annotationsBufs)
    if (importAnnotationsRaw(info.getFIRLoc(), annotationsBuf->getBuffer(),
                             annos))
      return failure();

  parseAnnotationTimer.stop();

  // Get annotations that are supposed to be specially handled by the
  // LowerAnnotations pass.
  if (!annos.empty())
    circuit->setAttr(rawAnnotations, b.getArrayAttr(annos));

  // A timer to get execution time of module parsing.
  auto parseTimer = ts.nest("Parse modules");
  deferredModules.reserve(16);

  // Parse any contained modules.
  while (true) {
    switch (getToken().getKind()) {
    // If we got to the end of the file, then we're done.
    case FIRToken::eof:
      goto DoneParsing;

    // If we got an error token, then the lexer already emitted an error,
    // just stop.  We could introduce error recovery if there was demand for
    // it.
    case FIRToken::error:
      return failure();

    default:
      emitError("unexpected token in circuit");
      return failure();

    case FIRToken::kw_class:
    case FIRToken::kw_declgroup:
    case FIRToken::kw_domain:
    case FIRToken::kw_extclass:
    case FIRToken::kw_extmodule:
    case FIRToken::kw_intmodule:
    case FIRToken::kw_layer:
    case FIRToken::kw_formal:
    case FIRToken::kw_module:
    case FIRToken::kw_option:
    case FIRToken::kw_public:
    case FIRToken::kw_simulation:
    case FIRToken::kw_type: {
      auto indent = getIndentation();
      if (!indent.has_value())
        return emitError("'module' must be first token on its line"), failure();
      unsigned definitionIndent = *indent;

      if (definitionIndent <= circuitIndent)
        return emitError("module should be indented more"), failure();

      if (parseToplevelDefinition(circuit, definitionIndent))
        return failure();
      break;
    }
    }
  }

  // After the outline of the file has been parsed, we can go ahead and parse
  // all the bodies.  This allows us to resolve forward-referenced modules and
  // makes it possible to parse their bodies in parallel.
DoneParsing:
  // Each of the modules may translate source locations, and doing so touches
  // the SourceMgr to build a line number cache.  This isn't thread safe, so we
  // proactively touch it to make sure that it is always already created.
  (void)getLexer().translateLocation(info.getFIRLoc());

  // Pre-verify symbol table, so we can construct it next.  Ideally, we would do
  // this verification through the trait.
  { // Memory is tight in parsing.
    // Check that all symbols are uniquely named within child regions.
    DenseMap<Attribute, Location> nameToOrigLoc;
    for (auto &op : *circuit.getBodyBlock()) {
      // Check for a symbol name attribute.
      auto nameAttr =
          op.getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName());
      if (!nameAttr)
        continue;

      // Try to insert this symbol into the table.
      auto it = nameToOrigLoc.try_emplace(nameAttr, op.getLoc());
      if (!it.second) {
        op.emitError()
            .append("redefinition of symbol named '", nameAttr.getValue(), "'")
            .attachNote(it.first->second)
            .append("see existing symbol definition here");
        return failure();
      }
    }
  }

  SymbolTable circuitSymTbl(circuit);

  moduleFixups.resize(deferredModules.size());

  // Stub out inner symbol namespace for each module,
  // none should be added so do this now to avoid walking later
  // to discover that this is the case.
  for (auto &d : deferredModules)
    innerSymbolNamespaces.get(d.moduleOp.getOperation());

  // Next, parse all the module bodies.
  auto anyFailed = mlir::failableParallelForEachN(
      getContext(), 0, deferredModules.size(), [&](size_t index) {
        if (parseModuleBody(circuitSymTbl, deferredModules[index],
                            moduleFixups[index]))
          return failure();
        return success();
      });
  if (failed(anyFailed))
    return failure();

  // Walk operations created that have inner symbol references
  // that need replacing now that it's safe to create inner symbols everywhere.
  for (auto &fixups : moduleFixups) {
    if (failed(fixups.resolve(innerSymbolNamespaces)))
      return failure();
  }

  // Helper to transform a layer name specification of the form `A::B::C` into
  // a SymbolRefAttr.
  auto parseLayerName = [&](StringRef name) -> Attribute {
    // Parse the layer name into a SymbolRefAttr.
    auto [head, rest] = name.split(".");
    SmallVector<FlatSymbolRefAttr> nestedRefs;
    while (!rest.empty()) {
      StringRef next;
      std::tie(next, rest) = rest.split(".");
      nestedRefs.push_back(FlatSymbolRefAttr::get(getContext(), next));
    }
    return SymbolRefAttr::get(getContext(), head, nestedRefs);
  };

  auto getArrayAttr = [&](ArrayRef<std::string> strArray, auto getAttr) {
    SmallVector<Attribute> attrArray;
    auto *context = getContext();
    for (const auto &str : strArray)
      attrArray.push_back(getAttr(str));
    if (attrArray.empty())
      return ArrayAttr();
    return ArrayAttr::get(context, attrArray);
  };

  if (auto enableLayers =
          getArrayAttr(getConstants().options.enableLayers, parseLayerName))
    circuit.setEnableLayersAttr(enableLayers);
  if (auto disableLayers =
          getArrayAttr(getConstants().options.disableLayers, parseLayerName))
    circuit.setDisableLayersAttr(disableLayers);

  auto getStrAttr = [&](StringRef str) -> Attribute {
    return StringAttr::get(getContext(), str);
  };

  if (auto selectInstChoice =
          getArrayAttr(getConstants().options.selectInstanceChoice, getStrAttr))
    circuit.setSelectInstChoiceAttr(selectInstChoice);

  circuit.setDefaultLayerSpecialization(
      getConstants().options.defaultLayerSpecialization);

  return success();
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

// Parse the specified .fir file into the specified MLIR context.
mlir::OwningOpRef<mlir::ModuleOp>
circt::firrtl::importFIRFile(SourceMgr &sourceMgr, MLIRContext *context,
                             mlir::TimingScope &ts, FIRParserOptions options) {
  auto sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  SmallVector<const llvm::MemoryBuffer *> annotationsBufs;
  unsigned fileID = 1;
  for (unsigned e = options.numAnnotationFiles + 1; fileID < e; ++fileID)
    annotationsBufs.push_back(
        sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID() + fileID));

  context->loadDialect<CHIRRTLDialect>();
  context->loadDialect<FIRRTLDialect, hw::HWDialect>();

  // This is the result module we are parsing into.
  mlir::OwningOpRef<mlir::ModuleOp> module(ModuleOp::create(
      FileLineColLoc::get(context, sourceBuf->getBufferIdentifier(),
                          /*line=*/0,
                          /*column=*/0)));
  SharedParserConstants state(context, options);
  FIRLexer lexer(sourceMgr, context);
  if (FIRCircuitParser(state, lexer, *module, minimumFIRVersion)
          .parseCircuit(annotationsBufs, ts))
    return nullptr;

  // Make sure the parse module has no other structural problems detected by
  // the verifier.
  auto circuitVerificationTimer = ts.nest("Verify circuit");
  if (failed(verify(*module)))
    return {};

  return module;
}

void circt::firrtl::registerFromFIRFileTranslation() {
  static mlir::TranslateToMLIRRegistration fromFIR(
      "import-firrtl", "import .fir",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        mlir::TimingScope ts;
        return importFIRFile(sourceMgr, context, ts);
      });
}
