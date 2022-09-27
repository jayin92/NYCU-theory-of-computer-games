all:
	g++ -std=c++11 -O3 -g -Wall -fmessage-length=0 -o threes threes.cpp
stats:
	./threes --total=1000 --save=stats.txt
judge: all stats
	cp stats.txt ../judge/
	../judge/threes-judge --load=stats.txt
clean:
	rm threes