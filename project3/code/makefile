all:
	g++ -std=c++17 -O3 -g -pthread -Wall -fmessage-length=0 -o nogo nogo.cpp
judge:
	cp nogo ../judge
	cd ../judge; ./run-gogui-twogtp.sh 2
test:
	./nogo --total=100 --black="parallel=4 T=1500 search=MCTS simulation=50000" --white="parallel=40 T=1500 search=MCTS simulation=0" --block=1 --limit=1 --save=stats.txt
clean:
	rm nogo
