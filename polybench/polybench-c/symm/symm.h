/**
 * symm.h: This file is part of the PolyBench/C 3.2 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */
#ifndef SYMM_H
# define SYMM_H

/* Default to STANDARD_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define STANDARD_DATASET
# endif

/* Do not define anything if the user manually defines the size. */
# if !defined(NI) && !defined(NJ)
/* Define the possible dataset sizes. */
#  ifdef MINI_DATASET
#   define NI 32
#   define NJ 32
#  endif

#  ifdef SMALL_DATASET
#   define NI 128
#   define NJ 128
#  endif

#  ifdef STANDARD_DATASET /* Default if unspecified. */
#   define NI 1024
#   define NJ 1024
#  endif

#  ifdef LARGE_DATASET
#   define NI 2000
#   define NJ 2000
#  endif

#  ifdef EXTRALARGE_DATASET
#   define NI 4000
#   define NJ 4000
#  endif
# endif /* !N */

# define _PB_NI POLYBENCH_LOOP_BOUND(NI,ni)
# define _PB_NJ POLYBENCH_LOOP_BOUND(NJ,nj)

# ifndef DATA_TYPE
#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.2lf "
# endif


#endif /* !SYMM */
