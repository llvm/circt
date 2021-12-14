#!/bin/sh
! "$1" "$3" 2>&1 | grep "$2" >/dev/null
