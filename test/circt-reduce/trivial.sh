#!/bin/sh
! firtool "$1" 2>&1 | grep "error: sink \"x1.x\" not fully initialized" >/dev/null
