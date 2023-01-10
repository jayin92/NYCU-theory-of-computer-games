all:
	g++ -std=c++17 -O3 -g -pthread -Wall -fmessage-length=0 -o nogo nogo.cpp
judge:
	cp nogo ../judge
	cd ../judge; ./run-gogui-twogtp.sh 8
test:
	./nogo --total=100 --black="parallel=4 T=1500" --white="T=100" --block=1 --limit=1 --save=stats.txt
clean:
	rm nogo
