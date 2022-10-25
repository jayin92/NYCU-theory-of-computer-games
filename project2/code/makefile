all:
	g++ -std=c++11 -O3 -g -Wall -fmessage-length=0 -o threes threes.cpp
stats:
	./threes --total=1000 --save=stats.txt
train:
	-rm weights.bin
	./threes --total=0 --slide="init=$weights_size save=weights.bin" # generate a clean network
	# ./threes --total=100000 --block=1000 --limit=1000 --slide="init=65536,65536,65536,65536,65536,65536,65536,65536 save=weights.bin alpha=0.0025"
	./threes --total=100000 --block=1000 --limit=1000 --slide="init=16777216,16777216,16777216,16777216 save=weights.bin alpha=0.0045"
	./threes --total=1000 --slide="load=weights.bin alpha=0" --save="stats.txt"
judge: clean all train
	cp stats.txt ../judge/
	../judge/threes-judge --load=stats.txt --judge="version=2"
clean:
	-rm threes