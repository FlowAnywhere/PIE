#!/bin/bash
cd /PIE/PIE
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
tensorflow_model_server  --port=9000 --model_name=pie --model_base_path=/PIE/output/export/BestExport/ &> ../output/serving.log &
#flask run --host=0.0.0.0 --port=19999 &> ../output/web_server.log &
gunicorn -k gevent -w 1 -b 0.0.0.0:19999 wsgi &> ../output/web_server.log &
