#!/bin/bash

cd extracted/ende/

for lang in de en; do
	for f in *.$lang; do
	    mv -- "$f" "${f%}.txt"
	done
done
