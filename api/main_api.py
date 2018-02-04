from flask import Flask, render_template
app = Flask(__name__)


@app.route("/")
def profile():
    return(render_template("api/template.html"))

if __name__=="__main_":
    app.run()