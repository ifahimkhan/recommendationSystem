from flask import *
from flask import render_template
from recsystem import get_recommendations,get_Suggetion
import pandas as pd
from flask_cors import CORS

f = Flask(__name__)
CORS(f) 

@f.route("/")   #this is the root folder 
def home():  
    return render_template("Search.html")  #This is to make call to html file

@f.route("/search")
def searchData():
    querry=request.args.get("q")
    list=get_recommendations(querry)
    df=pd.DataFrame(list)
    df.reset_index(inplace=True)
    a=df["movie_title"]
    print(a)
    return render_template("userList.html",tables=df.to_html())

@f.route('/autocomp',methods=['GET'])
def autocomplete():
    #return render_template("autocomp.html")
    querry=request.args.get("q")
    newList=[]
    for i in get_Suggetion() :
     if check(i.lower(),querry.lower()):
      newList.append(i)
    return Response(json.dumps(newList), mimetype='application/json')

def check(string, sub_str): 
    if (string.find(sub_str) == -1): 
        return False
    else: 
        return True

if __name__=='__main__':
    f.run(debug=True)

