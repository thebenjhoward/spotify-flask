from flask import Flask, request, make_response
import utils
app = Flask(__name__)

@app.route('/predict')
def song():
    uri = request.args.get('id', default="", type=str)
    if(uri == ""):
        res = make_response("No URI given", 400)
        res.mimetype = "test/plain"
        return res
    else:
        try:
            label, real = utils.classify_song(uri)
            res = make_response({'pred_label': label, 'real_label' : real}, 200)
            res.mimetype = "application/json"
            return res
        except Exception:
            res = make_response("Unknown Error", 500)
            res.mimetype = "test/plain"
            return res

if(__name__ == "__main__"):
    app.run(threaded=True)