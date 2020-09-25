// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Base class for other typedefs. Provides dialact-specific defaults
class ESI_Type<string name> : TypeDef<ESI_Dialect, name> { }

def FixedPoint : ESI_Type<"FixedPoint"> {
    let mnemonic = "fixed";

    let parameters = (
        ins
        "bool": $isSigned,
        "size_t": $wholeBits,
        "size_t": $fractionalBits
    );
}

def FloatingPoint : ESI_Type<"FloatingPoint"> {
    let mnemonic = "float";

    let parameters = (
        ins
        "bool": $isSigned,
        "size_t": $expontentBits,
        "size_t": $mantissaBits
    );
}

def List : ESI_Type<"List"> {
  let mnemonic = "list";

  let parameters = (
    ins
    "Type": $inner
  );
}

def String : ESI_Type<"String"> {
  let mnemonic = "string";
}

def MessagePointer : ESI_Type<"MessagePointer"> {
  let mnemonic = "ptr";

  let parameters = (
    ins
    "Type": $inner
  );
}

class FieldInfo_Type<string name> : ESI_Type<name> {
  let parameters = (
    ins
    ArrayRefOfSelfAllocationParameter<"::mlir::esi::FieldInfo", "">: $fields
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
    if (parser.parseLess()) return Type();
    while (mlir::succeeded(parser.parseOptionalLBrace())) {
      StringRef name;
      if (parser.parseKeyword(&name)) return Type();
      if (parser.parseComma()) return Type();
      Type type;
      if (parser.parseType(type)) return Type();
      if (parser.parseRBrace()) return Type();
      parameters.push_back(FieldInfo(name, type));
      if (parser.parseOptionalComma()) break;
    }
    if (parser.parseGreater()) return Type();
    return get(ctxt, parameters);
  }];
}

def StructType : FieldInfo_Type<"Struct"> {
  let mnemonic = "struct";
}

def UnionType : FieldInfo_Type<"Union"> {
  let mnemonic = "union";
}

def EnumType : ESI_Type<"Enum"> {
  let mnemonic = "enum";

  let parameters = (
    ins
    ArrayRefParameter<"::llvm::StringRef", "">: $fields
  );
}

def ESI_Type : Type<CPred<"$_self.getDialect().getTypeID() == TypeID::get<ESIDialect>()">, "ESI type">;
