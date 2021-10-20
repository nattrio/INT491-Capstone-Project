from utils.preprocess import *

class experiment:
    def __init__(self, feature, target, df):
        self.feature = feature
        self.target = target
        self.filtered_df = dropRecords(df, feature)
        self.classified_df = self.filtered_df
        
    def divideMethod(self, col_list, method, n_class):
        toClassMethod(col_list, method, n_class, self.classified_df)
    
    def divideCustom(self, col_list, scale):
        toClassCustom(col_list, scale, self.classified_df) 
        