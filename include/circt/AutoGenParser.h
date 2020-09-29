//===- AutoGenParser.h - standard format printer/parser ----------*- C++
//-*-===//
//
// Automatic (template-base) generation of parsers and printers. Defines a
// standard format for type asm format.
//
//===----------------------------------------------------------------------===//

#ifndef __AUTOGENPARSER_H__
#define __AUTOGENPARSER_H__

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
// These structs handle Type parameters' printing for common types
//
//===----------------------------------------------------------------------===//

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

template <typename T>
void PrintAll(mlir::DialectAsmPrinter &printer, T t) {
  Print<T>::go(printer, t);
}

template <typename T, typename... Args>
void PrintAll(mlir::DialectAsmPrinter &printer, T t, Args... args) {
  Print<T>::go(printer, t);
  printer << ", ";
  PrintAll(printer, args...);
}

template <typename... Args>
struct ParseAll {};

template <typename T>
struct ParseAll<T> {
  Parse<T> ParseHere;
  T result;

  template <typename Type, typename... ConArgs>
  mlir::Type construct(mlir::MLIRContext *ctxt, ConArgs... args) {
    return Type::get(ctxt, args..., result);
  }

  mlir::ParseResult go(mlir::MLIRContext *ctxt, mlir::DialectAsmParser &parser,
                       llvm::ArrayRef<llvm::StringRef> fieldNames) {
    if (ParseHere.go(ctxt, parser, fieldNames[0], result))
      return mlir::failure();
    return mlir::success();
  }
};

template <typename T, typename... Args>
struct ParseAll<T, Args...> {
  ParseAll<Args...> ParseNext;

  Parse<T> ParseHere;
  T result;

  template <typename Type, typename... ConArgs>
  mlir::Type construct(mlir::MLIRContext *ctxt, ConArgs... args) {
    return ParseNext.template construct<Type>(ctxt, args..., result);
  }

  mlir::ParseResult go(mlir::MLIRContext *ctxt, mlir::DialectAsmParser &parser,
                       llvm::ArrayRef<llvm::StringRef> fieldNames) {
    if (ParseHere.go(ctxt, parser, fieldNames[0], result))
      return mlir::failure();
    if (parser.parseComma())
      return mlir::failure();
    if (ParseNext.go(ctxt, parser, fieldNames.slice(1)))
      return mlir::failure();
    return mlir::success();
  }
};

} // namespace autogen

template <typename Type, typename... Args>
mlir::Type AutogenParser(mlir::MLIRContext *ctxt,
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

template <typename... Args>
void AutogenPrinter(mlir::DialectAsmPrinter &printer, llvm::StringRef name,
                    Args... args) {
  printer << name << "<";
  autogen::PrintAll(printer, args...);
  printer << ">";
}

} // namespace circt

#endif // __AUTOGENPARSER_H__
