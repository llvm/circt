//===- Types.td - FIRRTL Types Definition ------------------*- tablegen -*-===//
//
// This describes the FIRRTL types.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FIRRTL Types Definitions
//===----------------------------------------------------------------------===//

def FIRRTLType : Type<CPred<"$_self.isa<FIRRTLType>()">, "FIRRTLType">;

def ClockType : Type<CPred<"$_self.isa<ClockType>()">, "clock">,
                BuildableType<"ClockType::get($_builder.getContext())">;

def IntType : Type<CPred<"$_self.isa<IntType>()">, "sint or uint type">;
def SIntType : Type<CPred<"$_self.isa<SIntType>()">, "sint type">;
def UIntType : Type<CPred<"$_self.isa<UIntType>()">, "uint type">;

def UInt1Type : Type<CPred<"$_self.isa<UIntType>() && "
                           "$_self.cast<UIntType>().getWidth() == 1">,
                           "UInt<1>">,
                BuildableType<"UIntType::get($_builder.getContext(), 1)">;

def ResetType : Type<CPred<"$_self.isa<FIRRTLType>() && "
                           "$_self.cast<FIRRTLType>().isResetType()">,
                    "Reset, AsyncReset, or UInt<1>">;

def PassiveType : Type<CPred<"$_self.isa<FIRRTLType>() && "
                             "$_self.cast<FIRRTLType>().isPassiveType()">,
                       "a passive type (contain no flips)">;

def UIntSIntClockType : AnyTypeOf<[SIntType, UIntType, ClockType],
                                  "sint, uint, or clock">;

//===----------------------------------------------------------------------===//
// FIRRTL Enum Definitions
//===----------------------------------------------------------------------===//

/// RUWAttr: Undefined/Old/New.
def RUW_Undefined: I32EnumAttrCase<"Undefined", 0>;
def RUW_Old: I32EnumAttrCase<"Old", 1>;
def RUW_New: I32EnumAttrCase<"New", 2>;

def RUWAttr: I32EnumAttr<"RUWAttr", "Read Under Write Enum",
                         [RUW_Undefined, RUW_Old, RUW_New]>;

def MemDir_Infer    : I32EnumAttrCase<"Infer", 0>;
def MemDir_Read     : I32EnumAttrCase<"Read", 1>;
def MemDir_Write    : I32EnumAttrCase<"Write", 2>;
def MemDir_ReadWrite: I32EnumAttrCase<"ReadWrite", 3>;

def MemDirAttr: I32EnumAttr<"MemDirAttr", "Memory Direction Enum",
                            [MemDir_Infer, MemDir_Read, MemDir_Write,
                              MemDir_ReadWrite]>;
