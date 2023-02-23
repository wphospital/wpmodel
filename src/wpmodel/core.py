from abc import ABC, abstractmethod
from wpconnect.wpapi import WPAPIRequest
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

class WPModel(ABC):
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
    get_data(query_list : list = None)
        gets data as defined in the query_list

    """
    
    def __init__(
        self,
        model_name : str = strings.DEFAULT_MODEL_NAME,
        **kwargs
    ):
        self.model_name = model_name
        
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
        
        req = WPAPIRequest(wpapi_password)
        
        res = req\
            .get(
                query_fn,
                **kwargs
            )
        
        return res.get_data() if res.status_code == 200 else res
    
    @log
    def get_data(
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
        
        for q in query_list:
            key = self._get_query_key_name(q['query_fn'])

            self.data_dict[key] = self._get_wpapi_data(**q)
            
    @abstractmethod
    def preprocess(self):
        pass
    
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass