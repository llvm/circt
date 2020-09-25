// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

class ESI_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<ESI_Dialect, mnemonic, traits>;


// def ESI_CastCompound: ESI_Op<"cast_compound", [NoSideEffect]> {
//     let arguments = (ins ESI_Type:$value);
//     let results = (outs AnyI1:$result);
// }
