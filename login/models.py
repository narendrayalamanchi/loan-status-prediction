from django.db import models

# Create your models here.

class Login(models.Model): 
    id = models.AutoField(primary_key=True)
    useremail = models.CharField(db_column='userEmail', unique=True, max_length=100) 
    name = models.CharField(max_length=100)
    password = models.CharField(db_column='Password', max_length=500)
    lastlogindate = models.DateTimeField(db_column='lastLoginDate', blank=True, null=True)
    createdon = models.DateTimeField(db_column='createdOn', blank=True, null=True)
    loginattempts = models.IntegerField(db_column='loginAttempts', blank=True, null=True)
    randomkey = models.CharField(max_length=45) 

class historyofloanstatus(models.Model): 
    id = models.AutoField(primary_key=True)
    user = models.ForeignKey(Login, on_delete=models.CASCADE) 
    loan_amount = models.IntegerField(db_column='loan_amount')
    term = models.IntegerField(db_column='term')
    credit_score = models.IntegerField(db_column='credit_score')
    annual_income = models.IntegerField(db_column='annual_income')
    years_in_current_job = models.IntegerField(db_column='years_in_current_job')
    home_ownership = models.IntegerField(db_column='home_ownership')
    purpose = models.IntegerField(db_column='purpose')
    monthly_debt = models.IntegerField(db_column='monthly_debt')
    years_of_credit_history = models.IntegerField(db_column='years_of_credit_history')
    number_of_open_accounts = models.IntegerField(db_column='number_of_open_accounts')
    number_of_credit_problems = models.IntegerField(db_column='number_of_credit_problems')
    current_credit_balance = models.IntegerField(db_column='current_credit_balance')
    maximum_open_credit = models.IntegerField(db_column='maximum_open_credit')
    bankruptcies = models.IntegerField(db_column='bankruptcies')
    tax_liens = models.IntegerField(db_column='tax_liens')
    predictvalue = models.IntegerField(db_column='predictvalue', blank=True, null=True)
    