#REPOSITORY                  TAG                 IMAGE ID            CREATED             SIZE
#flowanywhere/pie            0.0.4               77d800ef5c83        5 minutes ago       1.77GB
docker run -v /git/PIE:/PIE -p 19999:19999 -it 77d800ef5c83 bash

export PYTHONPATH=/PIE
cd /PIE/PIE
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
tensorflow_model_server  --port=9000 --model_name=pie --model_base_path=/PIE/output/export/BestExport/ &> ../output/serving.log &
#flask run --host=0.0.0.0 --port=19999 &> ../output/web_server.log &
gunicorn -k gevent -w 1 -b 0.0.0.0:19999 app &> ../output/web_server.log &


docker-machine ip # get external IP address of docker on windows