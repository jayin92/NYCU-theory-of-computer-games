#!/bin/bash
P1B='./nogo --shell --name="Hollow-Black" --black="mcts T=1000"'
P1W='./nogo --shell --name="Hollow-White" --white="mcts T=1000"'

P2B='./nogo-judge --shell --name="Judge-Weak-Black" --black="weak"'
P2W='./nogo-judge --shell --name="Judge-Weak-White" --white="weak"'

./run-gogui-twogtp.sh 20 # play 20 games in total
