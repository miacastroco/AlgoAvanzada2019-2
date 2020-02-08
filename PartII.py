from collections import OrderedDict
import time

from quantopian.algorithm import attach_pipeline, pipeline_output, order_optimal_portfolio
from quantopian.pipeline import Pipeline, CustomFactor
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.optimize import TargetWeights
import random
from quantopian.pipeline.factors import Returns

import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB

num_holding_days = 5
days_for_fundamentals_analysis = 20
upper_percentile = 20
lower_percentile = 30

MAX_GROSS_EXPOSURE = 1.0
MAX_POSITION_CONCENTRATION = 0.05

def initialize(context):
    set_commission(commission.PerShare(cost=0.001, min_trade_cost=0))
    set_asset_restrictions(security_lists.restrict_leveraged_etfs)

    schedule_function(rebalance, date_rules.week_start(), time_rules.market_open(minutes=1))

    attach_pipeline(make_pipeline(), 'my_pipeline')

class Predictor(CustomFactor):
    
    def sigmoid(self, val):
        return 1.0 / (1 + np.exp(-val))
    
    def initializeClass(self, nodosVisibles, nodosOcultos):

        
        self.nodosOcultos = nodosOcultos
        self.nodosVisibles = nodosVisibles
        self.verb = True
        np_rng = np.random.RandomState(7860)


        self.pesos = np.asarray(np_rng.uniform(
                    low = -4 * np.sqrt(6. / (nodosOcultos + nodosVisibles)),
                    high = 4 * np.sqrt(6. / (nodosOcultos + nodosVisibles)),
                    size = (nodosVisibles, nodosOcultos)))


        self.pesos = np.insert(self.pesos, 0, 0, axis = 0)
        self.pesos = np.insert(self.pesos, 0, 0, axis = 1)


    
    
    tmpData = OrderedDict([
              ('Asset_Growth_2d' , Returns(window_length=2)), ('Asset_Growth_3d' , Returns(window_length=3)),
              ('Asset_Growth_4d' , Returns(window_length=4)), ('Asset_Growth_5d' , Returns(window_length=5)),
              ('Asset_Growth_6d' , Returns(window_length=6)), ('Asset_Growth_7d' , Returns(window_length=7)),
              ('Asset_Growth_8d' , Returns(window_length=8)), ('Asset_Growth_9d' , Returns(window_length=9)),
              ('Asset_Growth_10d' , Returns(window_length=10)), ('Asset_Growth_15d' , Returns(window_length=15)),
              ('Asset_Growth_10d' , Returns(window_length=10)), ('Asset_Growth_20d' , Returns(window_length=20)),
              ('Return' , Returns(inputs=[USEquityPricing.open],window_length=5))
              ])
    
    def entrenar(self, data, intentosMaximos, factorAprendizaje = 0.08):

        ejemplos = data.shape[0]
        data = np.insert(data, 0, 1, axis = 1)

        T0 = 1000
        T = T0

        for epoch in range(intentosMaximos):

            activacionesPositivas = np.dot(data, self.pesos)/T
            probPositivas = self.sigmoid(activacionesPositivas)
            probPositivas[:,0] = 1

            estadosPositivos = probPositivas > np.random.rand(ejemplos,
                            self.nodosOcultos + 1)
            asociacionesPositivas = np.dot(data.T, probPositivas)


            activacionesNegativasVis = np.dot(estadosPositivos, self.pesos.T)/T
            probNegativasVis = self.sigmoid(activacionesNegativasVis)
            probNegativasVis[:,0] = 1

            activacionesNegativasOcul = np.dot(probNegativasVis, self.pesos)/T
            probNegativasOcul = self.sigmoid(activacionesNegativasOcul)
            asociacionesNegativas = np.dot(probNegativasVis.T, probNegativasOcul)

            self.pesos += factorAprendizaje * ((asociacionesPositivas -
                            asociacionesNegativas) / ejemplos)
            error = np.sum((data - probNegativasVis) ** 2)
            
            T = T0/(1 + np.log(epoch))
            epoch += 1
            if self.verb:
                print('Iteración %s: Error es: %s' % (epoch, error))

    

    columns = list(tmpData.keys())
    inputs = list(tmpData.values())


    def compute(self, today, assets, out, *inputs):

        inputs = OrderedDict([(self.columns[i] , pd.DataFrame(inputs[i]).fillna(0,axis=1).fillna(0,axis=1)) for i in range(len(inputs))])
        num_secs = len(inputs['Return'].columns)
        y = inputs['Return'].shift(-num_holding_days)
        y=y.dropna(axis=0,how='all')
        
        for index, row in y.iterrows():
            
             upper = np.nanpercentile(row, upper_percentile)            
             lower = np.nanpercentile(row, lower_percentile)
             auxrow = np.zeros_like(row)
             
             for i in range(0,len(row)):
                if row[i] <= lower: 
                    auxrow[i] = -1
                elif row[i] >= upper: 
                    auxrow[i] = 1 
        
             y.iloc[index] = auxrow
            
        y=y.stack(dropna=False)
        
        x = pd.concat([df.stack(dropna=False) for df in list(inputs.values())], axis=1).fillna(0)
        
        model = GaussianNB() 
        model_x = x[:-num_secs*(num_holding_days)]
        model.fit(model_x, y)
        
        out[:] =  model.predict(x[-num_secs:])

def make_pipeline():

    universe = QTradableStocksUS()

    pipe = Pipeline(columns={'Model': Predictor(window_length=days_for_fundamentals_analysis, mask=universe)},screen = universe)

    return pipe

def rebalance(context,data):

    pipeline_output_df = pipeline_output('my_pipeline').dropna(how='any')
    
    todays_predictions = pipeline_output_df.Model

    target_weight_series = todays_predictions.sub(todays_predictions.mean())

    target_weight_series = target_weight_series*random.random()/target_weight_series.abs().sum()
    
    order_optimal_portfolio(objective=TargetWeights(target_weight_series),constraints=[])
