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
        # data from inference asset 
        output = self.data['output']
        proba = self.data['probability']
        
        # output file save for re-train later (mandatory)
        output_path = self.asset.get_output_path() # needed: .csv only 1 / .jpg only 1 / .only, .jpg each 1
        output.to_csv(output_path + 'output.csv')

        # inference summary (mandatory)
        num_survived = len(output[output['predicted']==1])
        num_total = len(output)
        survival_ratio = num_survived / num_total 
        # 1: survived / 0: dead
        avg_proba = np.average(proba, axis=0)
        avg_proba_survived = avg_proba[1].item() # float
        avg_proba_dead = avg_proba[0].item()  
        
        if num_total > 300:
            note = "sample_data1 used for inference"
        elif 100 < num_total <= 300:
            note = "sample_data2 used for inference"
        else:
            note = "sample_data3 used for inference"
 
        
        self.asset.save_info("\nALO test -- start !!!")
        self.asset.save_summary(result=f"#survived:{num_survived} / #total:{num_total}", \
               score=round(survival_ratio, 3), \
               note=note, \
               probability={"dead":avg_proba_dead, "survived":avg_proba_survived})       
        
        self.asset.save_data(self.data)
        self.asset.save_config(self.config)
 
 
#--------------------------------------------------------------------------------------------------------------------------
#    MAIN
#--------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    envs, argv, data, config = {}, {}, {}, {}
    ua = UserAsset(envs, argv, data, config)
    ua.run()
