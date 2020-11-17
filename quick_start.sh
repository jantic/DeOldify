#!/bin/bash
function usage {
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
	echo "usage : $0 notebook password -- to start the notebook with password"
	echo "             leave empty for no password (not recommended)"
	echo "usage : $0 image_api  -- to start image api"
	echo "usage : $0 video_api  -- to start video api"
	echo ''
	echo 'you can add non mandatory arguments'
	echo "usage : $0 image_api $port $host -- for custom port or host"
	echo ''
	echo ''
}

NOTEBOOK_PASSWORD=$2

if [ "$1" == "" ]; then
	echo "missing first argument"
	usage 
else
	docker run -d -p 8888:8888 -p 5000:5000 -e NOTEBOOK_PASSWORD=$NOTEBOOK_PASSWORD deoldify run_$1 || docker build -t deoldify -f Dockerfile . && docker run -it -p 8888:8888 -p 5000:5000 -e NOTEBOOK_PASSWORD=$NOTEBOOK_PASSWORD deoldify run_$1
fi
    