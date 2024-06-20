#!/bin/bash

while read -r id; do
  echo $id
  # Turn on after making sure it works
  # gh cache delete $id
  sleep 2
done
