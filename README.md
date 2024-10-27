1. Install pip: Check the link below to install the pip package.
https://pip.pypa.io/en/stable/installation/

2. Unzip the project source folder 
   cd loan-status-prediction → Go inside loan-status-prediction folder and follow the instructions below by creating virtual environment and running migrations.

3. Creating Virtual Environment and installing requirements inside Virtual Environment:
    Install virtual environment in your machine : pip3 install virtualenv
    Create virtual environment : virtualenv env -p python3
    (env folder will be created by running the command.)
    Activate virtual environment : source env/bin/activate
    I have listed all the python packages that are needed for the project inside requirements.txt.
    Install requirements: pip3 install -r requirements.txt 
4. Running Sqlite Database Migrations:
    All of my models are inside login app :python3 manage.py makemigrations login
    (After applying makemigrations, the migrations folder will be created. )
    python3 manage.py migrate


5. After completed all the steps from the above run the application using below command:
	python3 manage.py runserver (By default Django will be running in port 8000)

