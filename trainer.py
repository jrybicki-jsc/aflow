import mlflow
from sklearn.linear_model import LogisticRegression
import mlflow.sklearn
import numpy as np
import sys
import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd
import os

def setup_data(db_url='sqlite:////tmp/data.db'):
   engine = create_engine(db_url, echo=True)
   conn = engine.connect()
   conn.execute('CREATE table dat(x float, value float)')

   for i in range(-2, 2, 1):
      print(i, 1 if i>0 else 0)
      conn.execute(f'INSERT into dat(x, value) values({i},{1 if i>0 else 0} )')
   conn.close()

def get_data(db_url='sqlite:////tmp/data.db'):
   engine = create_engine(db_url, echo=True)
   conn = engine.connect()
   df =  pd.read_sql_table('data', conn)
   conn.close()
   return df

if __name__ == "__main__":
   print('Starting up...')
   #setup_data()

   with mlflow.start_run():
     mlflow.log_param("x", 1)
     mlflow.log_metric("y", 2)

     # metrics:
     for i in range(0, 3):
        mlflow.log_metric(key='quality', value=2*i, step=i)

     lr = LogisticRegression()
     df = get_data(db_url = os.environ.get('DB_URL', 'sqlite:////tmp/data.db'))
     X = df['x'].values.reshape(-1, 1)
     y = df['value'].values
     lr.fit(X, y)

     mlflow.sklearn.log_model(sk_model=lr, artifact_path='model', conda_env={
         'name': 'mlflow-env',
         'channels': ['conda-forge'],
         'dependencies': [
             'python=3.7.0',
             'scikit-learn=0.19.2',
             'pysftp=0.2.9',
             'pandas==0.25.1',
             'SQLAlchemy==1.3.8'
         ]
     })
