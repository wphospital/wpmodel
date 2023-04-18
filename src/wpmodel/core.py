# from abc import ABC, abstractmethod
from wpconnect.wpapi import WPAPIRequest, get_precache_list
from sprucepy import secrets

import logging
import time

import datetime as dt
import pytz

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

import hashlib
import yaml
import dill
import os
import json
import warnings
    
from . import strings
from . import constants

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

logger = logging.getLogger('WPModel Logger')
logger.setLevel(logging.INFO)
logger.propagate = False

def log(func):
    """Decorator for standardized logging
    """
    def wrapper(self, *args, **kwargs):
        try:
            start = time.time()
            result = func(self, *args, **kwargs)
            elapsed = time.time() - start
            
            logger.log(
                logging.INFO,
                '{} {} executed in {:.2f} seconds'.format(
                    self.__repr__(),
                    func.__name__,
                    elapsed
                )
            )
            return result
        except Exception as e:
            logger.exception(
                '{} raised exception in {}. exception: {}'.format(
                    self.__repr__(),
                    func.__name__,
                    str(e)
                )
            )
            raise e

    return wrapper

def fit(func):
    """Decorator for standard fitting ops
    """
    @log
    def fit(self, *args, **kwargs):
        result = func(self, *args, **kwargs)

        fitted_time = dt.datetime.now(tz=pytz.utc)

        if self.keep_fit_history:
            self.fit_history[fitted_time] = result

            self.model = self.fit_history[fitted_time]
        else:
            self.model = result
        
        self.fitted_time = fitted_time

        return result

    return fit

class WPModel:
    """
    A parent class for standardized model management at WPH.
    Designed for future model development to be created as a subclass
    
    TODO
    ----
    Implement standard logging
    

    Attributes
    ----------
    model_name : str
        optional - the name of the model. defaults to the default
        model name defined in strings.yml
    id : str
        hash value representing a unique ID for the class
    logfile : str
        optional - a log file name to log to
    created_time : datetime.datetime
        datetime objects with tzinfo=UTC. time the class was created
    fitted_time : datetime.datetime
        datetime object with tzinfo=UTC. time the model was fitted
    query_list : list
        a list of dictionaries with key-value pairs suitable for
        passing to wpconnect WPAPIRequest class
    data_dict : dict
        a dictionary with dataframes populated using the query_list
        and WPAPI

    Methods
    -------
    _configure_logger()
        configures the logger based on the optional class
        attribute logfile
    _set_id()
        creates the unique model id
    set_query_list(query_list : list)
        setter for the class attribute query_list
    _get_query_key_name(query_fn: str)
        converts the query_fn argument of WPAPI to a
        more human-readable dictionary key
    _get_wpapi_data(query_fn : str, **kwargs)
        uses the WPAPI interface to retrieve data
    load_data(query_list : list = None)
        gets data as defined in the query_list
    get_data(data_name, query_string)
        gets data from the data dictionary

    """
    
    def __init__(
        self,
        model_name : str = strings.DEFAULT_MODEL_NAME,
        keep_fit_history : bool = False,
        max_fit_history_size : int = 10,
        **kwargs
    ):
        self.model_name = model_name
        self.keep_fit_history = keep_fit_history
        self.max_fit_history_size = max_fit_history_size

        self.fit_history = {}
        
        self.logfile = kwargs.get('logfile', None)
        
        self._configure_logger()
        
        self.created_time = dt.datetime.now(tz=pytz.utc)
        self.fitted_time = None

        self._set_id()
        
        query_list = kwargs.get('query_list', [])
        
        self.set_query_list(query_list)
        self.data_dict = {}
        
    def __repr__(self):
        return '<{} {} {:%Y-%m-%d %H:%M:%S}>'.format(
            self.model_name,
            'Created',
            self.created_time
        )

    def _set_id(self):
        md5 = hashlib.md5(
            bytes(
                '{}{:%Y%m%d%H%M%S.%f}'.format(
                    self.model_name,
                    self.created_time
                ),
                'utf-8'
            )
        )

        self.id = md5.hexdigest()
    
    def _configure_logger(self):
        """Configures the logger
        """
        
        if (logger.hasHandlers()):
            logger.handlers.clear()

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if self.logfile:
            fh = logging.FileHandler(self.logfile)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    def save(
        self,
        filepath : str = 'out',
        clear_data : bool = True
    ):
        """Dump the whole shebang as a dill to the filepath

        Parameters
        ----------
        filepath : str
            the path to save the model to (the directory path, not filename)
        clear_data : bool
            boolean indicating whether data should be cleared before save
            defaults to true
        """

        save_name = '{}_{}.pkl'.format(
            self.model_name,
            self.created_time.strftime('%Y%m%d%H%M%S')
        )

        if clear_data:
            self.clear_data()

        with open(os.path.join(filepath, save_name), 'wb') as file: # TODO: Recursively make sure filepath exists
            dill.dump(self, file)
    
    def set_query_list(
        self,
        query_list : list
    ):
        """Sets the class attribute query_list
        
        Parameters
        ----------
        query_list : list
            a list of dictionaries with key-value pairs suitable for
            passing to wpconnect WPAPIRequest class 
        """
        
        self.query_list = query_list
    
    @staticmethod
    def _get_query_key_name(
        query_fn : str
    ):
        """Converts the query_fn argument of WPAPI to
        a more human-readable dictionary key
        
        Parameters
        ----------
        query_fn : str
            the name of a query file to run via WPAPI
        """
        
        return query_fn.replace('.sql', '')
    
    @staticmethod
    def _get_wpapi_data(
        query_fn : str,
        **kwargs
    ):
        """Uses the WPAPI interface to retrieve data
        
        Parameters
        ----------
        query_fn : str
            the name of the query to run
        kwargs
            other kwargs to pass along to WPAPI
            
        Returns
        -------
        pandas.DataFrame or WPAPIResponse
            the data retrieved from WPAPI (if a 200 response
            was received). otherwise, the WPAPIResponse object
        """
        
        wpapi_password = secrets.get_secret_by_key(
            'wpconnect_redis_password',
            api_url=constants.SPRUCE_API_URL
        )

        if query_fn == strings.MAGIC_DATA['WEATHER_QUERY_KEY']:
            endpoint = 'get_weather'
        else:
            endpoint = 'repo_query'
        
        req = WPAPIRequest(wpapi_password, endpoint=endpoint)
        
        res = req\
            .get(
                query_fn,
                **kwargs
            )
        
        return res.get_data() if res.status_code == 200 else res
    
    @log
    def load_data(
        self,
        query_list : list = None
    ):
        """Sets the class attribute data_dict by retrieving data
        as defined in the query_list
        
        Parameters
        ----------
        query_list : list
            a list of dictionaries with key-value pairs suitable for
            passing to wpconnect WPAPIRequest class
        """
        
        if query_list is None:
            query_list = self.query_list
            
        if len(query_list) == 0:
            raise Exception(strings.errors.NO_DATA_DICT)

        self._check_query_list(query_list)
        
        for q in query_list:
            key = self._get_query_key_name(q['query_fn'])

            self.data_dict[key] = self._get_wpapi_data(**q)

    @log
    def clear_data(
        self
    ):
        """Clears data from the data_dict
        """

        self.data_dict = {}
            
    @log
    def get_data(
        self,
        data_name : str,
        query_string : str = None
    ):
        """Gets data from the dictionary. Loads if not defined

        Parameters
        ----------
        data_name : str
            a string with the key for the data to retrieve from data_dict
        query_string : str
            optional Pandas query string to get a subset of data

        Returns
        -------
        pandas.DataFrame
            a dataframe with the selected data
        """

        if len(self.data_dict) == 0:
            self.load_data()

        try:
            if query_string:
                return self.data_dict[data_name]\
                    .copy()\
                    .query(query_string)
            else:
                return self.data_dict[data_name]
        except KeyError as err:
            raise Exception(strings.errors.NO_DATA_DEFINED)

    @log
    def set_target(
        self,
        data_name : str,
        target_col : str
    ):
        """Set the target dataframe and column

        Parameters
        ----------
        data_name : str
            a string with the key for the data to retrieve from data_dict
        target_col : str
            a string with the column in data_name to use as target
        """

        self.target = data_name, target_col

    @staticmethod
    def _check_query_list(
        query_list : list
    ):
        """Checks the query list for any non-precached queries and issues
        warnings about slowness if any are found
        
        Parameters
        ----------
        query_list : list
            a list of dictionaries with key-value pairs suitable for
            passing to wpconnect WPAPIRequest class
        """

        pc = get_precache_list()

        for q in query_list:
            qfn = q['query_fn']
            environ = q.get('environ', 'qa')

            relpc = pc.query('query_fn == @qfn & environ == @environ')

            if relpc.index.size == 0:
                if not qfn in strings.MAGIC_DATA.values():
                    warnings.warn(strings.errors.DATA_NOT_CACHED)
            else:
                check_params = sorted(
                    pair for pair in q['query_params'].items()
                )

                match = False
                for _, r in relpc.iterrows():
                    pc_params = sorted(
                        pair for pair in json.loads(r['params']).items()
                    )

                    match = pc_params == check_params

                    if match:
                        break

                if not match:
                    warnings.warn(strings.errors.DATA_NOT_CACHED)
