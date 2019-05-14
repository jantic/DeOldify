jupyter notebook --port=8888 --no-browser --allow-root --ip=0.0.0.0 --NotebookApp.token="" --NotebookApp.password="$(python set_password.py $NOTEBOOK_PASSWORD)"
