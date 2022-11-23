all:
	g++ -std=c++11 -O3 -g -Wall -fmessage-length=0 -o nogo nogo.cpp
judge:
	cp nogo ../judge
	cd ../judge; ./run-gogui-twogtp.sh 20
clean:
	rm nogo