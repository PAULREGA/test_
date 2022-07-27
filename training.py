
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from project import load_data, write_predictions

if __name__ == "__main__":
    ids, data, labels = load_data()
    clf = ElasticNet(random_state=0)
    clf.fit(data,labels)
    write_predictions(clf)
    
    
if __name__ == "__main__":
    ids, data, labels = load_data()
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(data,labels)
    write_predictions(clf)
    
    
    
if __name__ == "__main__":
    ids, data, labels = load_data()
    clf = Ridge(alpha=1.0)
    clf.fit(data,labels)
    write_predictions(clf)
    
    
    
    
if __name__ == "__main__":
    ids, data, labels = load_data()
    clf = RandomForestRegressor(random_state=1234)
    clf.fit(data,labels)
    write_predictions(clf) 
    
    
if __name__ == "__main__":
    ids, data, labels = load_data()
    clf = GradientBoostingRegressor(random_state=1234)
    clf.fit(data,labels)
    write_predictions(clf)    
