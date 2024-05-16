#asset_[step_name].py
 
# -*- coding: utf-8 -*-
import os
import sys
from alolib.asset import Asset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np 
import pandas as pd 
 
#--------------------------------------------------------------------------------------------------------------------------
#    CLASS
#--------------------------------------------------------------------------------------------------------------------------
class UserAsset(Asset):
    def __init__(self, asset_structure):
        super().__init__(asset_structure)
        self.args       = self.asset.load_args()
        self.config     = self.asset.load_config()
        self.data       = self.asset.load_data()
 
    @Asset.decorator_run
    def run(self):

        summary = self.asset.load_summary()
        # print(summary)
 
        summary['note'] = f"config dict: {self.config}"
 
        self.asset.save_summary(result = summary['result'], note = summary['note'][:100], score = summary['score'])
        
        self.asset.save_data(self.data)
        self.asset.save_config(self.config)
        
 
 
#--------------------------------------------------------------------------------------------------------------------------
#    MAIN
#--------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    envs, argv, data, config = {}, {}, {}, {}
    ua = UserAsset(envs, argv, data, config)
    ua.run()
