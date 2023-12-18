import os
import time

from flask import Flask, jsonify, request

from utils.logging_utils import set_logger
from utils.param import build_params
from predictor import Predictor

set_logger()
__file_dir__ = os.path.abspath(os.path.dirname(__file__))
print(f"当前server.py所在的文件夹:{__file_dir__}")

app = Flask(__name__)
predictor = Predictor(
    param=build_params(resume=True)
)


@app.route("/")
def index():
    return "欢迎使用命名实体识别模型!"


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    try:
        print(time.time())
        if request.method == 'GET':
            # Get请求的参数获取方式
            text = request.args.get('text', None)
        else:
            # POST请求的参数获取方式
            text = request.form.get('text', None)
        print(time.time())
        _r = predictor.predict(text=text)
        print(time.time())
        return jsonify(_r)
    except Exception as e:
        return jsonify({'code': 1, 'msg': f'服务器异常:{e}'})
