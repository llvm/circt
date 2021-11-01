#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Generated tablegen dialects expect to be able to find some symbols from
# the mlir.dialects package.
from mlir.dialects._ods_common import (
    _cext, segmented_accessor, equally_sized_accessor, extend_opview_class,
    get_default_loc_context, get_op_result_or_value, get_op_results_or_values)
