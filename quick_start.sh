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
}

NOTEBOOK_PASSWORD=$2

if [ "$1" == "" ]; then
	echo "missing first argument"
	usage 
else
	#handling non GPU accelerated hardware
	# if nvidia-smi not found => no ipc host
	# --ipc=host
	
	if nvidia-smi; then
		IPC_HOST="--ipc=host"
		echo "nvidia card found running on GPU"
	else
		IPC_HOST=""
		echo "No nvidia card found running on CPU"
	fi

	already_built=$(rm -rf /tmp/deoldify.built; if docker image ls | grep deoldify > /tmp/deoldify.built; then echo "ok"; else echo "not ok"; fi; rm -rf /tmp/deoldify.built)

	if [ "$already_built" == "ok" ]; then
		echo "Docker started from cache"
		echo "Access your $1 on port 5000 (api) or 8888 (notebook)"
		docker run -it -p 8888:8888 -p 5000:5000 -e NOTEBOOK_PASSWORD=$NOTEBOOK_PASSWORD deoldify run_$1
	else
		docker build -t deoldify -f Dockerfile . && docker run -it -p 8888:8888 -p 5000:5000 -e NOTEBOOK_PASSWORD=$NOTEBOOK_PASSWORD deoldify run_$1
		echo "Docker build and started"
		echo "Access your $1 on port 5000 (api) or 8888 (notebook)"
	fi
fi
    
