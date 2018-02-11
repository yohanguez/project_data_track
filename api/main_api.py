from flask import Flask, render_template
app = Flask(__name__)

#test
@app.route("/")
def profile():
    return(render_template("template.html"))

if __name__=="__main__":
    app.debug = True
    app.run()