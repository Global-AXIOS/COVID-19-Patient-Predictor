from flask import Flask,render_template,request
import pickle
# open a file, where you stored the pickled data
file = open('model.pkl', 'rb')
# dump information to that file
clf = pickle.load(file)
file.close()

app = Flask(__name__)

# pages declarations
@app.route("/",methods=["GET","POST"])
def home():
    if request.method == "POST":
        mydict=request.form
        age=int(mydict["age"])
        fatigue=int(mydict["fatigue"])
        diff_breating=int(mydict["diff_breating"])
        fever=int(mydict["fever"])
        runny_nose=int(mydict["runny_nose"])

        probablity=clf.predict_proba([[age,fever,fatigue,runny_nose,diff_breating]])[0][1]
        print(probablity)
        status="No Need To Worry or Consult Doctor Stay Home Stay Safe"
        if probablity>0.5:
            status="Consult Doctor as soon as possible! Don't Panic"
        return render_template("index.html",prob=probablity*100,status=status)
    return render_template("index.html")

# app run
if __name__ == "__main__":
    app.run(host = '0.0.0.0',debug = True)