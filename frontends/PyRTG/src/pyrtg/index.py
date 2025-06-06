#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .support import wrap_opviews_with_values
from .base.dialects import index

wrap_opviews_with_values(index, index.__name__)
