#!/bin/bash


best=0
filename=""
ls | grep .tar.gz | while read -r line; do
    tar -xvzf $line
    cp ./stats.txt ../judge/
    score=`../judge/threes-judge --load=stats.txt --judge="version=2" | grep Assessment | cut -d ' ' -f 2`
    if (( $(echo "$score > $best" | bc -l) )); then
        best=$score
        filename=$line
    fi
    echo "----"
    echo $line
    echo $score
    echo "---"
    echo $best
    echo $filename
done

echo $best

echo $filename
