all:
	g++ -std=c++11 -O3 -g -Wall -fmessage-length=0 -o threes threes.cpp
stats:
	./threes --total=1000 --save=stats.txt
train:
	./threes --total=100000 --block=1000 --limit=1000 --slide="init=65536,65536,65536,65536,65536,65536,65536,65536 save=weights.bin alpha=0.0025"
	./threes --total=1000 --slide="load=weights.bin alpha=0" --save="stats.txt"
judge: all train
	cp stats.txt ../judge/
	../judge/threes-judge --load=stats.txt --judge="version=2"
clean:
	rm threes