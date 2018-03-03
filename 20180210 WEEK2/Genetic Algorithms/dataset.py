# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 07:04:17 2016

@author: ironbar

Definition of the class dataset for handling the Santander competition dataset
"""


import numpy as np
import pandas as pd
import time
from sklearn import preprocessing
from ast import literal_eval


class SantanderDataset(object):
    """
    Class for storing the dataset of Santander competition
    It will give the data as it is requested
    """

    def __init__(self, dataset_root):
        """
        Loads the dataset
        """
        self.dataset_root = dataset_root
        self.__load_datasets(dataset_root)
        self.__prepare_datasets()
        #self.transform_dataset_for_training(self.df)


    def __load_datasets(self, dataset_root):
        """
        Loads all the datasets
        """
        dictionary_types = {
                            "sexo":'category',
                            "ult_fec_cli_1t":str,
                            "indresi":'category',
                            "indext":'category',
                            "indrel":'category',
                            "indfall":'category',
                            "nomprov":'category',
                            "segmento":'category',
                            "ind_empleado":'category',
                            "pais_residencia":'category',
                            "indrel":'category',
                            "antiguedad":np.int16,
                            "ind_nuevo":'category',
                            'indrel_1mes':'category',
                            'tiprel_1mes':'category',
                            'canal_entrada':'category',
                            "age":np.int8,
                            "ind_actividad_cliente":'category',
                            "ind_ahor_fin_ult1":np.int8,
                            "ind_aval_fin_ult1":np.int8,
                            "ind_cco_fin_ult1":np.int8,
                            "ind_cder_fin_ult1":np.int8,
                            "ind_cno_fin_ult1":np.int8,
                            "ind_ctju_fin_ult1":np.int8,
                            "ind_ctma_fin_ult1":np.int8,
                            "ind_ctop_fin_ult1":np.int8,
                            "ind_ctpp_fin_ult1":np.int8,
                            "ind_deco_fin_ult1":np.int8,
                            "ind_deme_fin_ult1":np.int8,
                            "ind_dela_fin_ult1":np.int8,
                            "ind_ecue_fin_ult1":np.int8,
                            "ind_fond_fin_ult1":np.int8,
                            "ind_hip_fin_ult1":np.int8,
                            "ind_plan_fin_ult1":np.int8,
                            "ind_pres_fin_ult1":np.int8,
                            "ind_reca_fin_ult1":np.int8,
                            "ind_tjcr_fin_ult1":np.int8,
                            "ind_valo_fin_ult1":np.int8,
                            "ind_viv_fin_ult1":np.int8,
                            "ind_nomina_ult1":np.int8,
                            "ind_nom_pens_ult1":np.int8,
                            "ind_recibo_ult1":np.int8,

                            "ind_ahor_fin_ult1_change":'category',
                            "ind_aval_fin_ult1_change":'category',
                            "ind_cco_fin_ult1_change":'category',
                            "ind_cder_fin_ult1_change":'category',
                            "ind_cno_fin_ult1_change":'category',
                            "ind_ctju_fin_ult1_change":'category',
                            "ind_ctma_fin_ult1_change":'category',
                            "ind_ctop_fin_ult1_change":'category',
                            "ind_ctpp_fin_ult1_change":'category',
                            "ind_deco_fin_ult1_change":'category',
                            "ind_deme_fin_ult1_change":'category',
                            "ind_dela_fin_ult1_change":'category',
                            "ind_ecue_fin_ult1_change":'category',
                            "ind_fond_fin_ult1_change":'category',
                            "ind_hip_fin_ult1_change":'category',
                            "ind_plan_fin_ult1_change":'category',
                            "ind_pres_fin_ult1_change":'category',
                            "ind_reca_fin_ult1_change":'category',
                            "ind_tjcr_fin_ult1_change":'category',
                            "ind_valo_fin_ult1_change":'category',
                            "ind_viv_fin_ult1_change":'category',
                            "ind_nomina_ult1_change":'category',
                            "ind_nom_pens_ult1_change":'category',
                            "ind_recibo_ult1_change":'category',
                            'product_buy':np.int8,
        }
        limit_rows   = 20000000
        start_time = time.time()
        self.eval_current = pd.read_csv(dataset_root + "eval_current_month_dataset.csv.gz",
                                   dtype=dictionary_types,
                                   nrows=limit_rows)
        print ('It took %i seconds to load the dataset' % (time.time()-start_time))
        start_time = time.time()
        self.eval_previous = pd.read_csv(dataset_root + "eval_previous_month_dataset.csv.gz",
                                   dtype=dictionary_types,
                                   nrows=limit_rows)
        print ('It took %i seconds to load the dataset' % (time.time()-start_time))
        self.train_current = pd.read_csv(dataset_root + "train_current_month_dataset.csv.gz",
                                   dtype=dictionary_types,
                                   nrows=limit_rows)
        print ('It took %i seconds to load the dataset' % (time.time()-start_time))
        start_time = time.time()
        self.train_previous = pd.read_csv(dataset_root + "train_previous_month_dataset.csv.gz",
                                   dtype=dictionary_types,
                                   nrows=limit_rows)
        print ('It took %i seconds to load the dataset' % (time.time()-start_time))
        print (len(self.eval_current), len(self.eval_previous))
        print (len(self.train_current), len(self.train_previous))
        return

    def __prepare_datasets(self, verbose=True):
        """
        Private function for modifying the datasets before training
        """
        for df in [self.train_current, self.eval_current]:
            #Discretize the data
            #print (list(range(20000, 200001, 10000)))
            renta_ranges = [0] + list(range(20000, 200001, 10000))
            renta_ranges += list(range(300000, 1000001, 100000)) + [2000000, 100000000]
            df.renta = pd.cut(df.renta, renta_ranges, right=True)
            # I'm going to use periods of one year
            antiguedad_ranges = [-10]+list(range(365, 7301, 365))+[8000]
            df.antiguedad = pd.cut(df.antiguedad, antiguedad_ranges, right=True)
            #age
            age_ranges = list(range(0, 101, 10))+[200]
            df.age = pd.cut(df.age, age_ranges, right=True)
            #Create month column
            df['month'] = (df.fecha_dato)%12 + 1
            df.month = df.month.astype('category')
        #Get column groups
        df = self.eval_previous
        change_columns = [name for name in df.columns if 'change' in name]
        product_columns = [name for name in df.columns
            if 'ult1' in name and 'change' not in name]
        df = self.eval_current
        categorical_columns = list( df.select_dtypes(
                                include=['category']).columns)
        #Create translation dictionary
        text = '{\n'
        df = self.eval_current
        for key in categorical_columns:
            text += '\t"%s":{' % key
            for i, category in enumerate(df[key].unique()):
                text += '"%s": %s,' % (category, i)
            text += ' },\n'
        #print (text)
        df = self.eval_previous
        for key in change_columns:
            text += '\t"%s":{' % key
            for i, category in enumerate(df[key].unique()):
                text += '"%s": %s,' % (category, i)
            text += ' },\n'
        text += '}\n'
        translation_dict = eval(text)
        print (translation_dict)
        print (df.describe())
        #Use the dictionary for translation
        for df in [self.train_current, self.eval_current]:
            for key in categorical_columns:
                if verbose: print (key)
                df[key].cat.categories = [translation_dict[key][str(category)]
                                        for category in  df[key].cat.categories]
        for df in [self.train_previous, self.eval_previous]:
            for key in change_columns:
                if verbose: print (key)
                df[key].cat.categories = [translation_dict[key][str(category)]
                                        for category in  df[key].cat.categories]
        print (df.describe())
        #Transform new_products column to a list
        df = self.eval_current
        df.new_products = df.new_products.apply(literal_eval)
        #Save some data for later
        self.change_columns = change_columns
        self.product_columns = product_columns
        self.categorical_columns = categorical_columns
        self.translation_dict = translation_dict


    def __get_encoded_data(self, df, input_columns):
        """
        Private method that uses one hot encoder for
        transforming the required data

        df: pandas dataframe
        input_columns: list with the names of the columns to use
        """
        #Get parameters for the encoder
        n_values = [len(self.translation_dict[key].values())
                    for key in input_columns]
        #Create the encoder
        enc = preprocessing.OneHotEncoder(n_values=n_values,
                                          sparse=False, dtype=np.uint8)
        #Fit the encoder
        enc.fit(df[input_columns].values)
        #Transform the data
        encoded_data = enc.transform(df[input_columns].values)
        return encoded_data

    def __get_data_aux(self, msg):
        """
        Auxiliary method for get_data
        It handles when there is more than one month requested
        """
        #Loop over the required months
        data = [None, None, None]
        for month in msg['month']:
            msg_copy = msg.copy()
            msg_copy['month'] = month
            ret = self.get_data(msg_copy)
            for i in range(3):
                #Aggregate with the data of other months if necessary
                if data[i] is None:
                    data[i] = ret[i]
                else:
                    data[i] = np.concatenate((data[i], ret[i]), axis=0)
        return data

    def get_data(self, msg, verbose=True):
        """
        Returns the data needed for training given the specified parameters

        The input is a message with the given fields

        month: int or list with the number of the month we want
            the data to be taken of
        train: bool, if true uses training dataset otherwise uses eval dataset
        input_columns: a list with the name of the columns we are going to use
            in the task
        use_product: bool, if true adds the product columns of the month before
        use_change: bool, if true adds the change columns of the month before
        """
        #TODO: I'm not filtering by month
        #TODO: The function for eval data will be very similar, try to reuse
        if verbose: print (msg)
        #If we have more than one month return aux function
        if type(msg['month']) is list:
            return self.__get_data_aux(msg)
        #Select the datasets we will be using
        if msg['train']:
            df_current = self.train_current[
                self.train_current.fecha_dato == msg['month']]
            df_previous = self.train_previous[
                self.train_previous.fecha_dato == msg['month']-1]
        else:
            #Then eval datasets are used
            df_current = self.eval_current[
                self.eval_current.fecha_dato == msg['month']]
            df_previous = self.eval_previous[
                self.eval_previous.fecha_dato == msg['month']-1]

        #Collect the required categorical data from current dataset
        input_data = None
        input_columns = msg['input_columns']
        if len(input_columns) > 0:
            #Get parameters for the encoder
            input_data = self.__get_encoded_data(df_current,
                                                 input_columns)
        #Collect the required data from previous dataset
        #Add product columns if necesary, product columns are binary
        if msg['use_product']:
            product_data = df_previous[self.product_columns].values
            if input_data is None:
                input_data = product_data
            else:
                #Join the matrixes
                if verbose: print (input_data.shape, product_data.shape)
                input_data = np.concatenate((input_data, product_data),
                                            axis=1)
        #Add change columns if necesary
        if msg['use_change']:
            change_data = self.__get_encoded_data(df_previous,
                                                     self.change_columns)
            if input_data is None:
                input_data = change_data
            else:
                #Join the matrixes
                if verbose: print (input_data.shape, change_data.shape)
                input_data = np.concatenate((input_data, change_data),
                                            axis=1)

        #Now collect the output data
        if msg['train']:
            output_data = df_current.buy_class.values
        else:
            output_data = df_current.new_products.values

        #Collect previous products data

        if msg['train']:
            previous_products = None
        else:
            previous_products = df_previous[self.product_columns].values

        return input_data, output_data, previous_products

