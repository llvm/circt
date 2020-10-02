//===- AutoGenParser.h - standard format printer/parser ---------*- C++-*-===//
//
// Automatic (template-base) generation of parsers and printers. Defines a
// standard format for type asm format.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_AUTOGENPARSER_H
#define CIRCT_AUTOGENPARSER_H

#include <mlir/IR/DialectImplementation.h>

namespace circt {
namespace autogen {

//===----------------------------------------------------------------------===//
//
// Template enables identify various types for which we have specializations
//
//===----------------------------------------------------------------------===//

template <typename...>
using void_t = void;

template <typename T>
using remove_constref =
    typename std::remove_const<typename std::remove_reference<T>::type>::type;

template <typename T, typename TestType>
using enable_if_type = typename std::enable_if<
    std::is_same<remove_constref<T>, TestType>::value>::type;

template <typename T, typename TestType>
using is_not_type =
    std::is_same<typename std::is_same<remove_constref<T>, TestType>::type,
                 typename std::false_type::type>;

template <typename T>
using get_indexable_type = remove_constref<decltype(std::declval<T>()[0])>;

template <typename T>
using enable_if_arrayref =
    enable_if_type<T, typename llvm::ArrayRef<get_indexable_type<T>>>;

//===----------------------------------------------------------------------===//
//
// These structs handle Type parameters' parsing for common types
//
//===----------------------------------------------------------------------===//

// This is the "interface" specification. It's a struct (vs a function) both
// because C++ doesn't allow partial template specification on function
// templates. Also, some parsers require temporary storage and structs provide a
// way to allocate stack space for that.
template <typename T, typename Enable = void>
struct Parse {
  mlir::ParseResult
  go(mlir::MLIRContext *ctxt,        // The context, should it be needed
     mlir::DialectAsmParser &parser, // The parser
     llvm::StringRef parameterName,  // Type parameter name, for
                                     // error printing (if necessary)
     T &result);                     // Put the parsed value here
};

// Int specialization
template <typename T>
using enable_if_integral_type =
    typename std::enable_if<std::is_integral<T>::value &&
                            is_not_type<T, bool>::value>::type;
template <typename T>
struct Parse<T, enable_if_integral_type<T>> {
  mlir::ParseResult go(mlir::MLIRContext *ctxt, mlir::DialectAsmParser &parser,
                       llvm::StringRef parameterName, T &result) {
    return parser.parseInteger(result);
  }
};

// Bool specialization -- 'true' / 'false' instead of 0/1
template <typename T>
struct Parse<T, enable_if_type<T, bool>> {
  mlir::ParseResult go(mlir::MLIRContext *ctxt, mlir::DialectAsmParser &parser,
                       llvm::StringRef parameterName, bool &result) {
    llvm::StringRef boolStr;
    if (parser.parseKeyword(&boolStr))
      return mlir::failure();
    if (!boolStr.compare_lower("false")) {
      result = false;
      return mlir::success();
    }
    if (!boolStr.compare_lower("true")) {
      result = true;
      return mlir::success();
    }
    llvm::errs() << "Parser expected true/false, not '" << boolStr << "'\n";
    return mlir::failure();
  }
};

// Float specialization
template <typename T>
using enable_if_float_type =
    typename std::enable_if<std::is_floating_point<T>::value>::type;
template <typename T>
struct Parse<T, enable_if_float_type<T>> {
  mlir::ParseResult go(mlir::MLIRContext *ctxt, mlir::DialectAsmParser &parser,
                       llvm::StringRef parameterName, T &result) {
    double d;
    if (parser.parseFloat(d))
      return mlir::failure();
    result = d;
    return mlir::success();
  }
};

// mlir::Type specialization
template <typename T>
using enable_if_mlir_type =
    typename std::enable_if<std::is_convertible<T, mlir::Type>::value>::type;
template <typename T>
struct Parse<T, enable_if_mlir_type<T>> {
  mlir::ParseResult go(mlir::MLIRContext *ctxt, mlir::DialectAsmParser &parser,
                       llvm::StringRef parameterName, T &result) {
    mlir::Type type;
    auto loc = parser.getCurrentLocation();
    if (parser.parseType(type))
      return mlir::failure();
    if ((result = type.dyn_cast_or_null<T>()) == nullptr) {
      parser.emitError(loc, "expected type '" + parameterName + "'");
      return mlir::failure();
    }
    return mlir::success();
  }
};

// llvm::StringRef specialization
template <typename T>
struct Parse<T, enable_if_type<T, llvm::StringRef>> {
  mlir::ParseResult go(mlir::MLIRContext *ctxt, mlir::DialectAsmParser &parser,
                       llvm::StringRef parameterName, llvm::StringRef &result) {
    mlir::StringAttr a;
    if (parser.parseAttribute<mlir::StringAttr>(a))
      return mlir::failure();
    result = a.getValue();
    return mlir::success();
  }
};

// ArrayRef specialization
template <typename T>
struct Parse<T, enable_if_arrayref<T>> {
  using inner_t = get_indexable_type<T>;
  Parse<inner_t> innerParser;
  llvm::SmallVector<inner_t, 4> parameters;

  mlir::ParseResult go(mlir::MLIRContext *ctxt, mlir::DialectAsmParser &parser,
                       llvm::StringRef parameterName,
                       llvm::ArrayRef<inner_t> &result) {
    if (parser.parseLSquare())
      return mlir::failure();
    if (failed(parser.parseOptionalRSquare())) {
      do {
        inner_t parameter; // = std::declval<inner_t>();
        innerParser.go(ctxt, parser, parameterName, parameter);
        parameters.push_back(parameter);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRSquare())
        return mlir::failure();
    }
    result = llvm::ArrayRef<inner_t>(parameters);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
//
// These structs handle data type parameters' printing for common types
//
//===----------------------------------------------------------------------===//

// This is the "interface" specification for ptiner. It's a struct (vs a
// function) because C++ doesn't allow partial template specification on
// function templates.
template <typename T, typename Enable = void>
struct Print {
  static void go(mlir::DialectAsmPrinter &printer, const T &obj);
};

// Several C++ types can just be piped into the printer
template <typename T>
using enable_if_trivial_print =
    typename std::enable_if<std::is_convertible<T, mlir::Type>::value ||
                            (std::is_integral<T>::value &&
                             is_not_type<T, bool>::value) ||
                            std::is_floating_point<T>::value>::type;
template <typename T>
struct Print<T, enable_if_trivial_print<remove_constref<T>>> {
  static void go(mlir::DialectAsmPrinter &printer, const T &obj) {
    printer << obj;
  }
};

// llvm::StringRef has to be quoted to match the parse specialization above
template <typename T>
struct Print<T, enable_if_type<T, llvm::StringRef>> {
  static void go(mlir::DialectAsmPrinter &printer, const T &obj) {
    printer << "\"" << obj << "\"";
  }
};

// bool specialization
template <typename T>
struct Print<T, enable_if_type<T, bool>> {
  static void go(mlir::DialectAsmPrinter &printer, const bool &obj) {
    if (obj)
      printer << "true";
    else
      printer << "false";
  }
};

// llvm::ArrayRef specialization
template <typename T>
struct Print<T, enable_if_arrayref<T>> {
  static void go(mlir::DialectAsmPrinter &printer,
                 const llvm::ArrayRef<get_indexable_type<T>> &obj) {
    printer << "[";
    for (size_t i = 0; i < obj.size(); i++) {
      Print<remove_constref<decltype(obj[i])>>::go(printer, obj[i]);
      if (i < obj.size() - 1)
        printer << ", ";
    }
    printer << "]";
  }
};

//===----------------------------------------------------------------------===//
//
// Print ALL the parameters in comma seperated form. These are functions since
// they can be.
//
//===----------------------------------------------------------------------===//

// Base case
template <typename T>
void printAll(mlir::DialectAsmPrinter &printer, T t) {
  Print<T>::go(printer, t);
}

// Recursive case
template <typename T, typename... Args>
void printAll(mlir::DialectAsmPrinter &printer, T t, Args... args) {
  Print<T>::go(printer, t);
  printer << ", ";
  printAll(printer, args...);
}

//===----------------------------------------------------------------------===//
//
// Parse ALL the parameters in comma seperated form. These are functions since
// they can be.
//
//===----------------------------------------------------------------------===//

// Handle a base case
template <typename... Args>
struct ParseAll {};

// Another base case
template <typename T>
struct ParseAll<T> {
  Parse<T> parseHere;
  T result;

  // Actually construct the type 'Type'. The argument list was constructed
  // recursively. This is the base case so it is the one to call Type::get(...).
  template <typename Type, typename... ConArgs>
  mlir::Type construct(mlir::MLIRContext *ctxt, ConArgs... args) {
    return Type::get(ctxt, args..., result);
  }

  // Parse this field and store it in 'result'
  mlir::ParseResult go(mlir::MLIRContext *ctxt, mlir::DialectAsmParser &parser,
                       llvm::ArrayRef<llvm::StringRef> fieldNames) {
    if (parseHere.go(ctxt, parser, fieldNames[0], result))
      return mlir::failure();
    return mlir::success();
  }
};

template <typename T, typename... Args>
struct ParseAll<T, Args...> {
  ParseAll<Args...> parseNext;

  Parse<T> parseHere;
  T result;

  // Construct the type 'Type'. Build the argument list recusively.
  template <typename Type, typename... ConArgs>
  mlir::Type construct(mlir::MLIRContext *ctxt, ConArgs... args) {
    return parseNext.template construct<Type>(ctxt, args..., result);
  }

  // Parse this field, putting the parsed value in 'result'. Then parse a comma
  // since there is another field after this one. (We are not the base case.)
  // Call the next one.
  mlir::ParseResult go(mlir::MLIRContext *ctxt, mlir::DialectAsmParser &parser,
                       llvm::ArrayRef<llvm::StringRef> fieldNames) {
    if (parseHere.go(ctxt, parser, fieldNames[0], result))
      return mlir::failure();
    if (parser.parseComma())
      return mlir::failure();
    if (parseNext.go(ctxt, parser, fieldNames.slice(1)))
      return mlir::failure();
    return mlir::success();
  }
};

} // namespace autogen

//===----------------------------------------------------------------------===//
//
// Outer-level functions to call from your parser / printer code
//
//===----------------------------------------------------------------------===//

// Generate a parser for a given list of types. Run the parser, construct the
// type 'Type', and return it. Most of the magic is above.
template <typename Type, typename... Args>
mlir::Type autogenParser(mlir::MLIRContext *ctxt,
                         mlir::DialectAsmParser &parser,
                         llvm::ArrayRef<llvm::StringRef> fieldNames) {
  autogen::ParseAll<Args...> parseAll;
  if (parser.parseLess())
    return mlir::Type();
  if (parseAll.go(ctxt, parser, fieldNames))
    return mlir::Type();
  if (parser.parseGreater())
    return mlir::Type();
  return parseAll.template construct<Type>(ctxt);
}

// Print a type given a list of types
template <typename... Args>
void autogenPrinter(mlir::DialectAsmPrinter &printer, llvm::StringRef name,
                    Args... args) {
  printer << name << "<";
  autogen::printAll(printer, args...);
  printer << ">";
}

//===----------------------------------------------------------------------===//
//
// Using the above outer-level functions requires a bit of redundancy. (Because
// C++ doesn't allow strings as template parameters.) These macros reduce this
// redundancy.
//
// There may be some stuff in c++17 or c++20 which reduce this redundancy.
// Ideally, we'd just specify the Type and compile time introspection would be
// used to gather the rest of this info.
//
//===----------------------------------------------------------------------===//

// Utility macros
#define GET_FIELD_TYPE(NAME, FIELD)                                            \
  decltype(std::declval<NAME>().getImpl()->FIELD)
#define GET_MACRO(_1, _2, _3, _4, _5, NAME, ...) NAME

#define APA1(CTXT, PAR, NAME, A)                                               \
  autogenParser<NAME, GET_FIELD_TYPE(NAME, A)>(CTXT, PAR, {#A})
#define APA2(CTXT, PAR, NAME, A, B)                                            \
  autogenParser<NAME, GET_FIELD_TYPE(NAME, A), GET_FIELD_TYPE(NAME, B)>(       \
      CTXT, PAR, {#A, #B})
#define APA3(CTXT, PAR, NAME, A, B, C)                                         \
  autogenParser<NAME, GET_FIELD_TYPE(NAME, A), GET_FIELD_TYPE(NAME, B),        \
                GET_FIELD_TYPE(NAME, C)>(CTXT, PAR, {#A, #B, #C})
#define APA4(CTXT, PAR, NAME, A, B, C, D)                                      \
  autogenParser<NAME, GET_FIELD_TYPE(NAME, A), GET_FIELD_TYPE(NAME, B),        \
                GET_FIELD_TYPE(NAME, C), GET_FIELD_TYPE(NAME, D)>(             \
      CTXT, PAR, {#A, #B, #C, #D})
#define APA5(CTXT, PAR, NAME, A, B, C, D, E)                                   \
  autogenParser<NAME, GET_FIELD_TYPE(NAME, A), GET_FIELD_TYPE(NAME, B),        \
                GET_FIELD_TYPE(NAME, C), GET_FIELD_TYPE(NAME, D),              \
                GET_FIELD_TYPE(NAME, E)>(CTXT, PAR, {#A, #B, #C, #D, #E})

// Use this macro: AUTOGEN_PARSER(ctxt, parser, paramName1, paramName2, ...).
// Supports up to 5 parameters (easy to add more).
#define AUTOGEN_PARSER(C, P, T, ...)                                           \
  return GET_MACRO(__VA_ARGS__, APA5, APA4, APA3, APA2, APA1)(C, P, T,         \
                                                              __VA_ARGS__)

#define APR1(printer, name, A) autogenPrinter(printer, #name, getImpl()->A);
#define APR2(printer, name, A, B)                                              \
  autogenPrinter(printer, #name, getImpl()->A, getImpl()->B);
#define APR3(printer, name, A, B, C)                                           \
  autogenPrinter(printer, #name, getImpl()->A, getImpl()->B, getImpl()->C)
#define APR4(printer, name, A, B, C, D)                                        \
  autogenPrinter(printer, #name, getImpl()->A, getImpl()->B, getImpl()->C,     \
                 getImpl()->D)
#define APR5(printer, name, A, B, C, D, E)                                     \
  autogenPrinter(printer, #name, getImpl()->A, getImpl()->B, getImpl()->C,     \
                 getImpl()->D, getImpl()->E)

// Use this macro AUTOGEN_PRINTER(printer, typeMnemonic paramName1, paramName2,
// ...). Supports up to 5 parameters (easy to add more).
#define AUTOGEN_PRINTER(P, N, ...)                                             \
  GET_MACRO(__VA_ARGS__, APR5, APR4, APR3, APR2, APR1)(P, N, __VA_ARGS__)

} // namespace circt

#endif // CIRCT_AUTOGENPARSER_H
