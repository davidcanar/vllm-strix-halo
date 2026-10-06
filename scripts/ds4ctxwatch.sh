# ctxwatch.sh LABEL SIZES -- context probe with a MemAvailable watcher on both boxes (min recorded)
L=$1; S=$2
( while true; do a1=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); a2=$(ssh -o BatchMode=yes 10.0.2.2 "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null); echo "$a1 $a2"; sleep 15; done ) > /tmp/ds4ctx_mem_$L.log 2>&1 &
W=$!
python3 -P "$(dirname "$0")/ds4ctx.py" $S 1 1234
kill $W 2>/dev/null
echo "   MemAvailable min during $L: box1 $(awk '{print $1}' /tmp/ds4ctx_mem_$L.log | sort -n | head -1) GB, box2 $(awk '{print $2}' /tmp/ds4ctx_mem_$L.log | sort -n | head -1) GB"
