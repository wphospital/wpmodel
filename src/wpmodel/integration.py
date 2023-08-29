# from multipledispatch import dispatch
import pandas as pd 
import numpy as np

from . import utils

class Integration:

    def __init__(
        self,
        models:list =[],
        **kwargs
    ):
        self.models = models
         
        
        if self.models:
            self.model_names = [m.model_name for m in self.models]
        elif kwargs.get('model_names',None):
            self.model_names = kwargs.get('model_names')
            if kwargs.get('model_repo', None):
                self.set_models(from_cloud=False,model_repo=kwargs.get('model_repo'))
            else:
                self.set_models(from_cloud=True)
            self.set_models(from_cloud=True)
        else:
            self.model_names= [i for i in utils.get_all_models(from_cloud=True) if 'cluster' not in i.lower()]
            self.set_models(from_cloud=True)
            
        
    def set_models(self,from_cloud=True,model_repo=''):
        # if self.check_model_exist():
        #     self.models=[]
        #     for i,*l in self.model_names:
        #         model = get_latest(i)
        #         model.set_fitted_model(model,l[0])
        #         self.models.append(model)
        if from_cloud:
            self.models = [utils.get_latest(i,from_cloud=True) for i in self.model_names]
        else:
            self.models = [utils.get_latest(i,from_cloud=False,model_df=model_repo) for i in self.model_names]
            
    def get_forecast(self,date_agg, start,end="2999-12-31", **kwargs):
        """ get forecast for each model in model list, and standardize the prediction/acutal column names
            

        Parameters
        ----------
        date_agg: str 
            date aggregation level - day,week,month,quarter
        start: str
            forecast start date
        end: str
            forecast end date
        """
        # change time frame to relative dates
        df_final = pd.DataFrame(columns=[date_agg])
        
        for m in self.models:
            string = {m.query_list[0]['query_fn'].strip('.sql'):f'{m.date_column}.between("{start}" , "{end}")'}
            # print(string)
            df = m.get_agg_prediction(
                string,
                date_agg,
                **kwargs
                )           
            
            df_final = pd.merge(df_final,df,how='outer').fillna(0)
            
        return self.add(df_final)
 
    def add(self,df):
        """add all prediction and acutual counts into `total`
        """
        columns = df.columns
        col_pred = [x for x in columns if 'pred_' in x and len(set(x.split('_')).intersection(set(self.model_names)))>0]
        col_actual = [x for x in columns if 'actual_' in x and len(set(x.split('_')).intersection(set(self.model_names)))>0]
        
        df['total_pred'] = df[col_pred].sum(axis=1)
        df['total_actual'] = df[col_actual].sum(axis=1)
        return df


    def multiply(self,model2,model1=None, query_string={}, df=None): 
        ## cluster model need to define date & percentage columns
        '''decompose total prediction into clusters
        
        Parameters
        ----------
        mode1:WPModel
        
        model2:Cluster model
        
        Return
        ------
        DataFrame of 
        '''
        ## standardized column name after aggregation
        ## ToDo: use overloading
        if df is None:
            try:
                model2.percent
            except AttributeError:
                model1,model2 = model2,model1
            df1 = self.get_pred_by_model(model1,query_string)
            df1['day'] = pd.to_datetime(df1['day'])
        else:
            df1 = df
        df2 = model2.predict()
        
        col = [x for x in df1.columns if 'pred_' in x]
        other = [x for x in df1.columns if x not in col]
        print(col)
        
        df = df1.merge(df2,left_on='day',right_on=model2.date_column,how='left') 
        df['cluster_total'] = df[col[0]] * df[model2.percent]
        
        return df
 

        
