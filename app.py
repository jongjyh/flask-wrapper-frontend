
from flask import Flask, request
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
from predictor import RewritePredictor
from data_reader import RewriteDatasetReader
from model import UnifiedFollowUp
import time 


# flask-swagger
from flask import Flask
from flask_restx import Api, Resource, fields,Namespace

app = Flask(__name__)
api = Api(app, version='0.1', title='Dialog Rewrite',
    description='debug some api',
)
ns = Namespace('test', description='test')
api.add_namespace(ns)

# predict wrapper
class PredictManager:

    def __init__(self, archive_file):
        archive = load_archive(archive_file)
        self.predictor = Predictor.from_archive(
            archive, predictor_name="rewrite")

    def predict_result(self, dialog_list):     
        #input: list  dialog_list = [utterance1, utterance2, ..., last_utterance]
        #output : str  last_utterance => new_utterance
        
        dialog_snippets = []
        for utterance in dialog_list:
            dialog_snippets.append(''.join(["%s " % i for i in utterance]))
        param = {
            "context": dialog_snippets[:-1],
            "current": dialog_snippets[-1]
        }
        restate = self.predictor.predict_json(param)["predicted_tokens"]
        return restate.replace(' ', '')
# loading model
manager = PredictManager("../pretrained_weights/rewrite_bert.tar.gz")
# inference
max_snippets = 8
def inference(dialog_list):
    last_user_snippet_idx = len(dialog_list)-1
    while last_user_snippet_idx>0 and ('user' not in dialog_list[last_user_snippet_idx] or dialog_list[last_user_snippet_idx]['user'] is ''):last_user_snippet_idx -= 1
    # 只关注最后一个有用户回复的对话序列，并截断过于遥远的对话
    dialog_list = dialog_list[:last_user_snippet_idx + 1]
    dialog_list = dialog_list[-max_snippets:]
    squeeze_list = []
    # 按system,user的顺序squeeze对话，确保system在最前，user随后
    for dialog in dialog_list:
        if 'system' in dialog:
            squeeze_list.append(dialog['system'])
        if 'user' in dialog:
            squeeze_list.append(dialog['user'])
    # 输入模型
    print(squeeze_list)
    start = time.time()
    pred = manager.predict_result(squeeze_list)
    elapsed = (time.time()-start)
    return pred,elapsed

# model definitions
snippet_fields = ns.model('Snippet',{
    'system': fields.String,
    'user': fields.String,
})
# 使用fields.Nested() 把里层的对象包装成数组
dialogs_fields = ns.model('Dialogs', {
    'dialog_history': fields.List(fields.Nested(snippet_fields),description=f'先前的system-user对话列表,从最后一个user字段不为空的对话开始，往前取{max_snippets}段对话。', required=True)
})

# post method
@ns.route('/pred')
class Prediction(Resource):
    @ns.expect(dialogs_fields) 
    def post(self):
        params = request.json
        dialog_list = params['dialog_history']
        pred_snippets,time = inference(dialog_list)
        data = {"rewritted_query":pred_snippets,
            "cost_time":round(time,5)}
        return data

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8889, debug=False)
