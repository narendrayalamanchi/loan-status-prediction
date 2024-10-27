from django.shortcuts import render,reverse
from django.http import HttpResponse,HttpResponseRedirect,JsonResponse
from .models import Login,historyofloanstatus
from .forms import historyofloanstatusForm
import random,string,json
from passlib.hash import bcrypt_sha256
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

# Create your views here.

def register(request):
    return render(request, 'register.html')


def registersuccess(request):
    context = {}

    if (request.method == 'POST'):
        context['error'] = False
        name = request.POST['name']
        useremail = request.POST['email']
        password = request.POST['password']
        tz = request.POST['tz']
        User_email_count = Login.objects.filter( useremail = useremail ).count()
        if User_email_count == 1:
            context['error'] = True
            context['errorMsg'] = "This email is already registered with us."
        date_now = timezone.now()

        encrypt_pswd = bcrypt_sha256.using(rounds=12).hash(password)
        userauth = Login(name=name , useremail=useremail, password=encrypt_pswd,createdon = date_now,loginattempts=0 )
        userauth.save()
        return JsonResponse(context)


def login(request):
    return render(request, 'login.html')

@csrf_exempt
def loginsuccess(request):
    email = request.POST['email']
    password = request.POST['password']
    context = {}
    try:
        name,userid, saltedpswd = Login.objects.values_list('name','id','password').get(useremail = email)
        if bcrypt_sha256.verify(password, saltedpswd):
            request.session['name'] = name
            request.session['id'] = userid
            # request.user = userid
            context['error'] = False
        else:
            context['error'] = True
            context['msg'] = "Invalid email or password"
    except Exception as e:
        print(repr(e))
        context['error'] = True
        context['msg'] = "Invalid email or password"

    return JsonResponse(context)

def dashboard(request):
    print("session",request.session['id'])
    return render(request, 'userdashboard.html') 

def checkloanstatus(request):
    if request.method == 'POST':
        request.POST._mutable = True
        for i in request.POST:
            if i != "csrfmiddlewaretoken":
                request.POST[i] = int(request.POST[i])

        form = historyofloanstatusForm(request.POST)
        request.POST["user"] = Login.objects.get(id=request.session['id']) 
        # check whether it's valid:
        try:
            if form.is_valid():
                print("Form is valid")
                predictedvalue = predictloanstatus(form.instance.loan_amount,form.instance.term,form.instance.credit_score,form.instance.annual_income,form.instance.years_in_current_job,form.instance.home_ownership,form.instance.purpose,form.instance.monthly_debt,form.instance.years_of_credit_history,form.instance.number_of_open_accounts,form.instance.number_of_credit_problems,form.instance.current_credit_balance,form.instance.maximum_open_credit,form.instance.bankruptcies,form.instance.tax_liens)
                form.instance.predictvalue = predictedvalue
                form.save()
                return render(request, 'userdashboard.html', {'form': form,"predictedvalue":predictedvalue})
        except Exception as e:
            print(repr(e))


    else:
        form = historyofloanstatusForm()

    return render(request, 'userdashboard.html', {'form': form})

def logout(request):
    idval = request.session['id']

    del request.session['id']
    del request.session['name']
    return HttpResponseRedirect(reverse('login:login'))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import sweetviz as sv

# Prediction Function
def predictloanstatus(loan_amount,term,credit_score,annual_income,years_in_current_job,home_ownership,purpose,monthly_debt,years_of_credit_history,number_of_open_accounts,number_of_credit_problems,current_credit_balance,maximum_open_credit,bankruptcies,tax_liens):
    train_dataset=pd.read_csv("train-dataset.csv")
    train_dataset.drop(columns = ["Loan ID","Customer ID"], inplace=True)
    train_dataset.drop_duplicates(inplace=True)
    train_dataset.drop(columns = ["Months since last delinquent"], inplace = True)
    # Dropping the rows if the row has more than 13 null values. It is unpredictable or guess with only 3 attributes/columns.
    train_dataset.dropna(thresh = 13, inplace = True)
    #Converting "Years in current job" object to integer by mapping.
    years = {
        "Years in current job": {
            "< 1 year": 0,
            "1 year": 1,
            "2 years": 2,
            "3 years": 3,
            "4 years": 4,
            "5 years": 5,
            "6 years": 6,
            "7 years": 7,
            "8 years": 8,
            "9 years": 9,
            "10+ years": 10   
        }
    }
    train_dataset['Years in current job']=train_dataset.replace(years)['Years in current job']
    #Filling/Replacing Null Values of columns/attributes
    train_dataset.interpolate(inplace=True)

    columns = ["Loan Status","Term","Home Ownership","Purpose"]
    for i in columns:
        encode = LabelEncoder()
        encode.fit(train_dataset[i])
        train_dataset[i] = encode.fit_transform(train_dataset[i])
    
    train = train_dataset[(np.abs(stats.zscore(train_dataset)) < 3).all(axis=1)]
    #Data Scaling
    Train_X = train.drop(['Loan Status'] , axis = 1).values
    Train_Y = train['Loan Status'].values

    analyze_report = sv.analyze(train.drop(['Loan Status'] , axis = 1))
    analyze_report.show_html('visualizations/train_report.html', open_browser=False)

    skscaler = StandardScaler()
    Train_X = skscaler.fit_transform(Train_X)

    TEST_X = np.array([[loan_amount,term,credit_score,annual_income,years_in_current_job,home_ownership,purpose,monthly_debt,years_of_credit_history,number_of_open_accounts,number_of_credit_problems,current_credit_balance,maximum_open_credit,bankruptcies,tax_liens]])
    test_df = pd.DataFrame(TEST_X, columns = ['loan_amount','term','credit_score','annual_income','years_in_current_job','home_ownership','purpose','monthly_debt','years_of_credit_history','number_of_open_accounts','number_of_credit_problems','current_credit_balance','maximum_open_credit','bankruptcies','tax_liens'])
    test_report = sv.analyze(test_df)
    test_report.show_html('visualizations/test_report.html', open_browser=False)

    compare_report = sv.compare([train.drop(['Loan Status'] , axis = 1), "Training Data"], [test_df, "Test Data"])
    compare_report.show_html('visualizations/traintestcomparison_report.html', open_browser=False)

    TEST_X = skscaler.fit_transform(TEST_X)
    
    # Logistic Regression is performed less accuracy w.r.t the validation set in previous phase.So, not including that in this phase.

    #Prediction with KNN 
    # knn = KNeighborsClassifier(n_neighbors=5)
    # knn.fit(Train_X,Train_Y)
    # knn_train = knn.score(Train_X,Train_Y)
    # print("KNN Train Score",knn_train)
    # y_knn_predicted = knn.predict(TEST_X)
    # print("KNN Predicted Value",y_knn_predicted[0])

    #Prediction with Desicion Tree
    dt = DecisionTreeClassifier( max_depth= 10 , max_features= 15)
    dt.fit(Train_X,Train_Y)
    dt_train = dt.score(Train_X,Train_Y)
    print("Desicion Tree Train Score",dt_train)
    y_dt_predicted = dt.predict(TEST_X)
    print("Desicion Tree Predicted Value",y_dt_predicted[0])

    #Prediction with Random Forest
    rf = RandomForestClassifier(n_estimators= 8 , max_depth=10 ,max_features=15) 
    rf.fit(Train_X,Train_Y)
    rf_train = rf.score(Train_X,Train_Y)
    print("Random Forest Train Score",rf_train)
    y_rf_predicted = rf.predict(TEST_X)
    print("Random Forest Predicted Value",y_rf_predicted[0])

    #Prediction with SVM 
    # svm_linear = svm.SVC(kernel = 'linear')
    # svm_linear.fit(Train_X,Train_Y)
    # svm_train = svm_linear.score(Train_X,Train_Y)
    # print("SVM Train Score",svm_train)
    # y_svm_predicted = svm_linear.predict(TEST_X)
    # print("SVM Predicted Value",y_svm_predicted[0])

    #Average of 4 models predicted value
    # avg = (y_knn_predicted[0]+y_dt_predicted[0]+y_rf_predicted[0]+y_svm_predicted[0])/4
    avg = (y_dt_predicted[0]+y_rf_predicted[0])/2
    print("Average",avg)
    #Returning round off value 
    return round(avg)
    

def history(request):
    userid = request.session['id']
    history = historyofloanstatus.objects.filter(user__id=userid).all()
    return render(request, 'history.html', {'history': history})

def trainreport(request):
    return render(request, 'train_report.html')

def testreport(request):
    return render(request, 'test_report.html')

def comparereport(request):
    return render(request, 'traintestcomparison_report.html')