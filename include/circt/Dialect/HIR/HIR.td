#ifndef HIR_OPS
#define HIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//Dialect
def HIRR_Dialect : Dialect {
    let name = "hir";
    let summary = "An MLIR dialect for Hardware acclerator description";
    let description = [{
      This dialect is for description of hardware accelerators with focus on
      FPGAs.
    }];
    let cppNamespace = "hir";
}




//Types
def HIR_TimeType : Type<CPred<"$_self.isa<TimeType>()">, "HIR time type">,
                    BuildableType<"TimeType::get($_builder.getContext())">;


//Ops
class HIR_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<HIRR_Dialect, mnemonic, traits>;

def HIR_Duplicate_Time_Var_Op : HIR_Op<"duplicate_time_var", [NoSideEffect]> {
    let summary = "Op duplicates a time var.";
    let description = [{
        This op duplicates a time var. Semantics are same as using original time
        var but in the hardware a separate path is used to propagate the new
        time var. 

        Example:

        ```mlir
        %t2 = hir.duplicate_time_var %t1: hir.time -> hir.time
        ```
    }];

    let arguments = (ins HIR_TimeType:$input);
    let results = (outs HIR_TimeType:$res);

    let assemblyFormat = [{$input attr-dict `:` type($input) `->` type($res)}];
}


def HIR_Def_Time_Var_Op : HIR_Op<"def_time_var", [NoSideEffect]> {
    let summary = "Op defines a new time var.";
    let description = [{
        This op defines a new time variable. 
        Example:

        ```mlir
        %t = hir.def_time_var: hir.time
        ```
    }];

    let arguments = (ins );
    let results = (outs HIR_TimeType:$res);

    let assemblyFormat = [{attr-dict `:` type($res)}];
}

#endif // HIR_OPS
