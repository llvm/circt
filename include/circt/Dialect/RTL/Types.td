//===- Types.td - RTL data type definitions ----------------*- tablegen -*-===//
//
// Basic data types for the RTL dialect.
//
//===----------------------------------------------------------------------===//

// Base class for other typedefs. Provides dialact-specific defaults
class RTLType<string name> : TypeDef<RTLDialect, name> { }

// A type with an array of `FieldInfo`s. Essentially, a parent class for the
// struct and union types.
class FieldInfo_Type<string name> : RTLType<name> {
  let parameters = (
    ins
    ArrayRefOfSelfAllocationParameter<"::circt::rtl::FieldInfo", "">: $fields
  );

  let printer = [{ printFieldInfoType($_printer, *this); }];
}

// A struct type. Inherits the parser/printer from FieldInfo_Type.
def StructType : FieldInfo_Type<"Struct"> {
  let summary = "A typical structure type";
  let description = [{
    A C-like or SystemVerilog-like struct. Missing alignment specification.
    We will add this once we figure out what exactly this means in hardware.
    (It may be obvious, we just don't want to make any assumptions.)
  }];
  let mnemonic = "struct";
  let parser = [{ return parseFieldInfoType<StructType>($_ctxt, $_parser); }];
}

// A union type. Inherits the parser/printer from FieldInfo_Type.
def UnionType : FieldInfo_Type<"Union"> {
  let summary = "A discriminated union";
  let description = [{
    Unlike a C union, stores a tag which indicates the field whose data is
    represented by this instance.
  }];
  let mnemonic = "union";
  let parser = [{ return parseFieldInfoType<UnionType>($_ctxt, $_parser); }];
}

// Fields which are an array of stringrefs need to re-allocate all the strings
// as well as the array itself.
class ArrayRefOfStringRef<string desc> :
  TypeParameter<"::llvm::ArrayRef<::llvm::StringRef>", desc> {

  let allocator = [{
    ::llvm::StringRef arr$_self [$_self.size()];
    for (size_t i = 0, e = $_self.size(); i < e; ++i)
      arr$_self[i] = $_allocator.copyInto($_self[i]);
    $_dst = $_allocator.copyInto(
      ::llvm::ArrayRef<::llvm::StringRef>(arr$_self, $_self.size()) );
  }];
}

// A simple enum is just a list of strings.
def EnumType : RTLType<"Enum"> {
  let summary = "A list of symbols";
  let description = [{
    A C-like or SystemVerilog-like enum. Just a list of strings for now. We
    will add value specifications if needed later on.
  }];

  let mnemonic = "enum";

  let parameters = (
    ins
    ArrayRefOfStringRef<"Possible enum values">: $fields
  );
}
