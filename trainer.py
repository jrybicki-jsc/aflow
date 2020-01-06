import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import mlflow.sklearn
import numpy as np
import sys
import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd
import os
import matplotlib.pyplot as plt


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
     mlflow.log_metric("y", 2)

     lr =  RandomForestRegressor(n_estimators=100)
     mlflow.log_param('n_estimators', lr.n_estimators)
     mlflow.log_param('min_samples_split', lr.min_samples_split)


     df = get_data(db_url = os.environ.get('DB_URL', 'sqlite:////tmp/mydata.db'))
     X = df['x'].values.reshape(-1, 1)
     y = df['value'].values
     lr.fit(X, y)

     mlflow.log_metric('score', lr.score(X, y))

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

     pred = lr.predict(X)
     plt.plot(X, pred, label='Prediction')
     plt.plot(X, y, label='Data')
     plt.legend()
     plt.savefig('predictions.png')
     mlflow.log_artifact('./predictions.png', 'performance')
