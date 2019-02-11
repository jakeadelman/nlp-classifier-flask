from flask import Flask
from sklearn.externals import joblib
import json
from flask import request
app = Flask(__name__)


@app.route('/post', methods=['POST'])
def hello():
    r = request.data
    r_parse = json.loads(r)
    clf = joblib.load('sent.plk')

    r = r_parse['data']
    # print(len(r), "HERE IS R LENGTH")
    i = 0
    newDict = {}
    while i < len(r):
        this = r[i]
        text = this['text']
        polarity = clf.predict([text])
        this['polarity'] = str(polarity[0])
        newDict[i] = this
        i += 1
    # print(newDict)
    json_dict = json.dumps(newDict)

    return json_dict
