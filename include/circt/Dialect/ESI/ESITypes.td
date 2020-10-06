//===- ESITypes.td - ESI data type definitions -------------*- tablegen -*-===//
//
// Data types for the ESI dialect
//
//===----------------------------------------------------------------------===//

// Base class for other typedefs. Provides dialact-specific defaults
class ESI_Type<string name> : TypeDef<ESI_Dialect, name> { }

def FixedPoint : ESI_Type<"FixedPoint"> {
  let summary = "ESI fixed point number type";
  let description = [{
    A fixed point number is an approximate representation of a real number --
    like a floating point number -- with the radix point fixed. While this
    representation lacks the dynamic range of floating point numbers, it is
    far simpler and more appropriate for many applications.

    It is typically represented as an `int` with the position of the radix
    point implictly define in the instructions. ESI supports this
    representation to not loose this type information. Operations on this
    type need not be any more complicated than on integer representations.
  }];

  let mnemonic = "fixed";
  let parser = [{ AUTOGEN_PARSER($_ctxt, $_parser, FixedPointType, isSigned, wholeBits, fractionalBits); }];
  let printer = [{ AUTOGEN_PRINTER($_printer, fixed, isSigned, wholeBits, fractionalBits); }];

  let parameters = (
    ins
    "bool": $isSigned,
    "size_t": $wholeBits,
    "size_t": $fractionalBits
  );
}

def FloatingPoint : ESI_Type<"FloatingPoint"> {
  let summary = "A floating point number with variable precision";
  let description = [{
    A typical floating point number with configurable signedness, exponent
    width, and mantissa size. Note: the semantics of operations on this type
    are not defined here. Certain operations may be IEEE754 compliant, some
    may not.
  }];

  let mnemonic = "float";
  let parser = [{ AUTOGEN_PARSER($_ctxt, $_parser, FloatingPointType, isSigned, exponentBits, mantissaBits); }];
  let printer = [{ AUTOGEN_PRINTER($_printer, float, isSigned, exponentBits, mantissaBits); }];

  let parameters = (
    ins
    "bool": $isSigned,
    "size_t": $exponentBits,
    "size_t": $mantissaBits
  );
}

def Array : ESI_Type<"Array"> {
  let summary = "A fixes-sized list of items";
  let description = [{
    Standard array parameterized by a type and the fixed number of items.
  }];

  let mnemonic = "array";
  let parser = [{ AUTOGEN_PARSER($_ctxt, $_parser, ArrayType, inner, size); }];
  let printer = [{ AUTOGEN_PRINTER($_printer, array, inner, size); }];

  let parameters = (
    ins
    "::mlir::Type": $inner,
    "size_t": $size
  );
}

def List : ESI_Type<"List"> {
  let summary = "A variably-sized list of items";
  let description = [{
    A typical ordered collection (list, c++ std::vector, etc.). There are a
    few different ways to represent this in hardware:

    - In memory, a linear array of packed items
    - On the wire (during transmit), a series of items transmitted N-items at
    a time wherein N is a parameter of the transmission channel.
  }];

  let mnemonic = "list";
  let parser = [{ AUTOGEN_PARSER($_ctxt, $_parser, ListType, inner); }];
  let printer = [{ AUTOGEN_PRINTER($_printer, list, inner); }];

  let parameters = (
    ins
    "::mlir::Type": $inner
  );
}


def ASCII : I32EnumAttrCase<"ASCII", 1>;
def UTF8  : I32EnumAttrCase<"UTF8",  2>;
def UTF16 : I32EnumAttrCase<"UTF16", 3>;
def UTF32 : I32EnumAttrCase<"UTF32", 4>;

def StringEncoding : I32EnumAttr<"StringEncoding",
    "Indicates the encoding of a string.",
    [ASCII, UTF8, UTF16, UTF32]> {

  let cppNamespace = "circt::esi";
}

def String : ESI_Type<"String"> {
  let summary = "A variably-sized string of characters";
  let description = [{
    Like a string in software, but is parameterized by the string encoding.
  }];

  let mnemonic = "string";
  let parser = [{ AUTOGEN_PARSER($_ctxt, $_parser, StringType, encoding); }];
  let printer = [{ AUTOGEN_PRINTER($_printer, string, encoding); }];

  let parameters = (
    ins
    "StringEncoding": $encoding
  );
}

def MessagePointer : ESI_Type<"MessagePointer"> {
  let summary = "A pointer to another place with the message";
  let description = [{
    Offset -- in **bits** relative to the location of this field -- to
    another field in the same message. Optionally can be negative to point to
    data which appeared previously. (Optional so that modules can declare
    that they don't have the ability to refer to earlier data -- that they
    don't buffer the entire message.)
    
    Not valid if location doesn't point to the beginning of an item of type
    `inner`. *Must* be recomputed if any modifications are made to the
    message (type, list length, etc.) which invalidate the pointer.

    Given the complexity, we don't anticipate this type being used often. It
    allows software to interface with (hopefully) less data marshalling. This
    type is subject to __massive__ changes as we explore hardware / software
    interfaces.
  }];

  let mnemonic = "ptr";
  let parser = [{ AUTOGEN_PARSER($_ctxt, $_parser, MessagePointerType, inner, negativeAllowed); }];
  let printer = [{ AUTOGEN_PRINTER($_printer, ptr, inner, negativeAllowed); }];

  let parameters = (
    ins
    "::mlir::Type": $inner,
    "bool": $negativeAllowed
  );
}

// A type with an array of `FieldInfo`s
class FieldInfo_Type<string name> : ESI_Type<name> {
  let parameters = (
    ins
    ArrayRefOfSelfAllocationParameter<"::circt::esi::FieldInfo", "">: $fields
  );

  let printer = [{
    printer << "struct" << "<";
    for (size_t i=0; i<getImpl()->fields.size(); i++) {
      const auto& field = getImpl()->fields[i];
      printer << "{" << field.name << "," << field.type << "}";
      if (i < getImpl()->fields.size() - 1)
        printer << ",";
    }
    printer << ">";
  }];

  let parser = [{
    SmallVector<FieldInfo, 4> parameters;
    if (parser.parseLess()) return ::mlir::Type();
    while (mlir::succeeded(parser.parseOptionalLBrace())) {
      StringRef name;
      if (parser.parseKeyword(&name)) return ::mlir::Type();
      if (parser.parseComma()) return ::mlir::Type();
      ::mlir::Type type;
      if (parser.parseType(type)) return ::mlir::Type();
      if (parser.parseRBrace()) return ::mlir::Type();
      parameters.push_back(FieldInfo(name, type));
      if (parser.parseOptionalComma()) break;
    }
    if (parser.parseGreater()) return ::mlir::Type();
    return get(ctxt, parameters);
  }];
}

def StructType : FieldInfo_Type<"Struct"> {
  let summary = "A typical structure type";
  let description = [{
    A C-like struct. Alignment to be defined in the future. Will need to
    define alignment on the wire and in memory. Given that alignment can be
    automatically modified (based on modules' requirements) it may be more
    appropriate to define the alignment in a 'data window'...
  }];
  let mnemonic = "struct";
}

def UnionType : FieldInfo_Type<"Union"> {
  let summary = "A discriminated union";
  let description = [{
    Unlike a C union, stores a tag which indicates the field whose data is
    represented by this instance.
  }];
  let mnemonic = "union";
}

def EnumType : ESI_Type<"Enum"> {
  let summary = "A list of symbols";
  let description = [{
    A C-like enum
  }];

  let mnemonic = "enum";
  let parser = [{ AUTOGEN_PARSER($_ctxt, $_parser, EnumType, fields); }];
  let printer = [{ AUTOGEN_PRINTER($_printer, enum, fields); }];

  let parameters = (
    ins
    ArrayRefParameter<"::llvm::StringRef", "">: $fields
  );
}

def ESI_Type : Type<CPred<"$_self.getDialect().getTypeID() == ::mlir::TypeID::get<ESIDialect>()">, "ESI type">;
