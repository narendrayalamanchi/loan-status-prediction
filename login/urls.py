from django.conf.urls import include,url
from . import views
app_name = 'login'

urlpatterns=[
    url(r'^$',views.register,name ='register'),
    url(r'^registersuccess$',views.registersuccess,name ='registersuccess'),

    url(r'^login$',views.login,name ='login'),
    url(r'^loginsuccess$',views.loginsuccess,name ='loginsuccess'),

    url(r'^dashboard$',views.dashboard,name ='dashboard'),
    url(r'^checkloanstatus$',views.checkloanstatus,name ='checkloanstatus'),
    url(r'^history$',views.history,name ='history'),

    url(r'^trainreport$',views.trainreport,name ='trainreport'),
    url(r'^testreport$',views.testreport,name ='testreport'),
    url(r'^traintestcomparisonreport$',views.comparereport,name ='comparereport'),
    
    url(r'^logout$',views.logout,name ='logout'),
]

