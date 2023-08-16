from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Root for home page


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
        data = CustomData(
            id=int(request.form.get('id')),
            year=int(request.form.get('year')),
            console=request.form.get('console'),
            category=request.form.get('category'),
            publisher=request.form.get('publisher'),
            rating=request.form.get('rating'),
            critics_points=float(request.form.get('critics_points')),
            user_points=float(request.form.get('user_points'))
        )
        print("Creating data as dataframe")
        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        print("predict results")
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=round(results[0], 2))


@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == 'POST':
        repo = git.Repo('./myproject')
        origin = repo.remotes.origin
        repo.create_head('master',
                         origin.refs.master).set_tracking_branch(origin.refs.master).checkout()
        origin.pull()
        return '', 200
    else:
        return '', 400


if __name__ == "__main__":
    app.run("0.0.0.0")
