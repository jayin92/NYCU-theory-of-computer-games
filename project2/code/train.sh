#!/bin/bash

weights_size="16777216,16777216,16777216,16777216" # 8x4-tuple
default=0.001
alpha="${1:-$default}"
for i in {1..20}; do
	echo "Step $i"
	./threes --total=75000 --block=1000 --limit=1000 --slide="load=weights.bin save=weights.bin alpha=${alpha}" | tee -a train.log
	./threes --total=1000 --slide="load=weights.bin alpha=0" --save="stats.txt"
	tar zcvf weights.$(date +%Y%m%d-%H%M%S).tar.gz weights.bin train.log stats.txt
	echo "Step $i finished"
done
