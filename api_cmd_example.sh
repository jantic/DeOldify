#!/bin/bash
echo '
  _____        ____  _     _ _  __       
 |  __ \      / __ \| |   | (_)/ _|      
 | |  | | ___| |  | | | __| |_| |_ _   _ 
 | |  | |/ _ \ |  | | |/ _` | |  _| | | |
 | |__| |  __/ |__| | | (_| | | | | |_| |
 |_____/ \___|\____/|_|\__,_|_|_|  \__, |
                                    __/ |
                                   |___/ 

'
echo "usage : $0 image  -- to test image api"
echo "usage : $0 video  -- to test video api"
echo ''
echo 'you can add non mandatory arguments'
echo "usage : $0 image port host -- for custom port or host"
echo ''
echo ''

if [ "$2" == "" ]; then
    port=5000
else
    port=$2
fi

if [ "$3" == "" ]; then
    host="127.0.0.1"
else
    host="$3"
fi

if [ "$1" == "video" ]; then
    echo "testing deOldify Video API on $host:$port"
    curl -X POST "http://$host:$port/process" -H "accept: application/octet-stream" -H "Content-Type: application/json" -d "{\"url\":\"https://v.redd.it/d1ku57kvuf421/HLSPlaylist.m3u8\", \"render_factor\":35}" --output colorized_video.mp4
elif [ "$1" == "image" ]; then
    echo "testing deOldify Image API on $host:$port"
    curl -X POST "http://$host:$port/process" -H "accept: image/png" -H "Content-Type: application/json" -d "{\"url\":\"http://www.afrikanheritage.com/wp-content/uploads/2015/08/slave-family-P.jpeg\", \"render_factor\":35}" --output colorized_image.png
fi

