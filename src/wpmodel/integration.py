# from multipledispatch import dispatch
import pandas as pd 
import numpy as np

from . import utils

class Integration:
 

    def __init__(
        self,
        models:list =[],
        pred_column:str = "total_pred",
        actual_column:str = "total_actual",
        **kwargs
    ):
        """Initialize model by providing
            a) a list of WPModels, 
            b) a list of WPModel names
                if no local model repo provided, will try to find a match in azure
            c) otherwise, use all available models except cluster model from azure  

        Parameters
        ----------
        models: a list of WPModels
        pred_column: column name of total predictions from sub components
        actual_column: column name of total actuals from sub components
        """
        self.models = models 
        self.pred_column = pred_column
        self.actual_column = actual_column
        
        if self.models:
            self.model_names = [m.model_name for m in self.models]

        elif kwargs.get('model_names',None):
            self.model_names = kwargs.get('model_names')
            if kwargs.get('model_repo', None):
                self.set_models_from_names(
                    self.model_names,
                    from_cloud=False,
                    model_repo=kwargs.get('model_repo')
                    )
            else:
                self.set_models_from_names(self.model_names,from_cloud=True)
        else:
            self.model_names= [i for i in utils.get_all_models(from_cloud=True)
                                if 'cluster' not in i.lower()]
            self.set_models_from_names(self.model_names, from_cloud=True)
     
    def set_models(
        self, 
        models:list =[]
    ):
        self.models = models
        self.model_names = [i.model_name for i in self.models]        
        
    def set_models_from_names(
        self,
        model_names: list=[], 
        from_cloud: bool=True,
        model_repo: str = ""
    ):
        ## ToDo: set previous versions
        if from_cloud:
            self.models = [utils.get_latest(i,from_cloud=True) for i in model_names]
        else:
            self.models = [utils.get_latest(i,
                from_cloud=False,
                model_df=model_repo)
                for i in model_names]
            
    def get_forecast(
        self,
        date_agg: str, 
        start: str,
        end: str = "2999-12-31",
        **kwargs
    ):
        """ get forecast for each model in model list, 
            and standardize the prediction/acutal column names           

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
            string = {m.query_list[0]['query_fn'].strip('.sql'):\
                f'{m.date_column}.between("{start}" , "{end}")'}
          
            df = m.get_agg_prediction(
                query_string=string,
                date_agg=date_agg,
                **kwargs
                )           
            
            df_final = pd.merge(df_final,df,how='outer').fillna(0)
            
        return self.add(df_final)
 
    def add(self,df):
        """add all prediction and acutual counts from components to get a total prediction & acutual
        """
        columns = df.columns
        col_pred = [x for x in columns if 'pred_' in x \
            and len(set(x.split('_')).intersection(set(self.model_names)))>0]
        col_actual = [x for x in columns if 'actual_' in x \
            and len(set(x.split('_')).intersection(set(self.model_names)))>0]
        
        df[self.pred_column] = df[col_pred].sum(axis=1)
        df[self.actual_column] = df[col_actual].sum(axis=1)
        return df


    def multiply(
        self,
        model2,
        model1=None, 
        query_string={}, 
        df=None
    ): 
        ## cluster model need to define date & percentage columns
        ## assuming df has column "day"

        '''decompose prediction & actual totals into clusters
        
        Parameters
        ----------
        mode1:WPModel
        model2:Cluster model
        
        Return
        ------
        DataFrame of 
        '''
        if df is None:
            try:
                model2.percent
            except AttributeError:
                model1,model2 = model2,model1
            df1 = model1.get_agg_prediction(query_string)   
        else:
            df1 = df

        df1['day'] = pd.to_datetime(df1['day'])
        df2 = model2.predict()
        
        col = [x for x in df1.columns if '_pred_' in x or '_actual_' in x ]
        if self.pred_column  in df1.columns:
            col.extend([self.pred_column,self.actual_column] )
        other = [x for x in df1.columns if x not in col]

        df = df1.merge(df2,left_on='day',right_on=model2.date_column,how='left') 

        for c in col:
            df[f'{c}_by_cluster'] = df[c] * df[model2.percent]
        
        return df