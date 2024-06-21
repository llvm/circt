#!/usr/bin/env bash

while read -r id; do
  gh cache delete $id
  sleep 2
done
