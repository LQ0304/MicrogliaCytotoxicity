import joblib

# read RF model
read_model_path = './data/FS_BM_RF_Results/ECFP_RFE/FS_BM_RF_ECFP_RFE7.pkl'
RF_Avalon_model = joblib.load(read_model_path)
print("max_depth:" + str(RF_Avalon_model.max_depth))
print("n_estimator:" + str(RF_Avalon_model.n_estimators))
print("max_features:" + str(RF_Avalon_model.max_features))

# read SVM model
read_model_path = './data/FS_BM_SVM_Results/ECFP_RFE/FS_BM_SVM_ECFP_RFE7.pkl'
SVM_model = joblib.load(read_model_path)
print("C:" + str(SVM_model.C))
print("shrinking:" + str(SVM_model.shrinking))
print("kernel:" + str(SVM_model.kernel))
print("gamma:" + str(SVM_model.gamma))

# read GBDT model
read_model_path = './data/FS_BM_GBDT_Results/ECFP_RFE/FS_BM_GBDT_ECFP_RFE7.pkl'
GBDT_model = joblib.load(read_model_path)
print("max_depth:" + str(GBDT_model.max_depth))
print("n_estimator:" + str(GBDT_model.n_estimators))
print("learning_rate:" + str(GBDT_model.learning_rate))
print("subsample:" + str(GBDT_model.subsample))
