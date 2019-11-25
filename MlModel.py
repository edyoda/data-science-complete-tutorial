from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from joblib import dump, load


class BuildMlPipeline:
    
    def __init__(self):
        pass
        
    def set_estimators(self, *args):
        estimator_db = {
            'randomForestRegressor': RandomForestRegressor(),
            'linearRegressor': LinearRegression(),
        }
        self.estimators = list(map( lambda algo: estimator_db[algo],args))
        
    def set_scalers(self, *args):
        scaler_db = {
            'standardscaler':StandardScaler(),
            'minmaxscaler':MinMaxScaler(),
        }
        self.scalers = list(map( lambda scaler: scaler_db[scaler],args))
        
    def set_samplers(self, *args):
        sampler_db = {
            'smote':SMOTE(),
            'smoteenn':SMOTEENN(),
        }
        self.samplers = list(map( lambda sampler: sampler_db[sampler],args))
        
    def set_encoders(self, *args):
        encoders_db = {
            'ohe':OneHotEncoder(handle_unknown='ignore'),
            'oe':OrdinalEncoder(),
        }
        self.encoders = list(map( lambda encoder: encoders_db[encoder],args))
        
    def set_hyperparameters(self, params):
        self.hyperparameters = params

    
    def create_pipelines(self, cat_cols, cont_cols):
        self.model_pipelines = []
        for scaler in self.scalers:
            pipeline_num = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                           ('scaling',scaler)])
            for encoder in self.encoders:
                pipeline_cat = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                               ('encoder',encoder)])
                preprocessor = make_column_transformer((pipeline_num, cont_cols),(pipeline_cat, cat_cols))
                
                for estimator in self.estimators:
                    pipeline  = make_pipeline(preprocessor, estimator)
                    self.model_pipelines.append(pipeline)
        

    def fit(self, trainX, trainY):
        self.gs_pipelines = []
        for idx,pipeline in enumerate(self.model_pipelines):
            elems = list(map(lambda x:x[0] ,pipeline.steps))
            param_grid = {}

            for elem in elems:
                if elem.lower() in self.hyperparameters:
                    param_grid.update(self.hyperparameters[elem])

            gs = GridSearchCV(pipeline, param_grid= param_grid, n_jobs=6, cv=5)
            gs.fit(trainX, trainY)
            print (gs.score(testX,testY),  list(map(lambda x:x[0] , gs.best_estimator_.steps)), gs.best_params_)

            dump(gs, 'model'+str(idx)+'.pipeline')
            self.gs_pipelines.append(gs)


    def score(self, testX, testY):
        for idx,model in enumerate(self.gs_pipelines):
            y_pred = model.best_estimator_.predict(testX)
            print (model.best_estimator_)
            print (idx,confusion_matrix(y_true=testY,y_pred=y_pred))
