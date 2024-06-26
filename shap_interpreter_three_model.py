import os
import numpy as np
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import xlwt

'''function:shap analysis three models'''
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 14

Top_N = 15
# load non FS
'''6种分子表征形式：不同分子描述符'''
repre_list = ['Avalon', 'ECFP', 'FCFP', 'Rdkit']
s = repre_list[1]
load_data_path = './data/save_data_BM/ECFP/'
train_x_origin = np.load(os.path.join(load_data_path, s + '_' + 'train_x7.npy'))
train_y_origin = np.load(os.path.join(load_data_path, s + '_' + 'train_y7.npy'))
test_x_origin = np.load(os.path.join(load_data_path, s + '_' + 'test_x7.npy'))
test_y_origin = np.load(os.path.join(load_data_path, s + '_' + 'test_y7.npy'))
x_data_origin = np.array(np.concatenate([train_x_origin, test_x_origin], axis=0), dtype='int')
y_data_origin = np.array(np.concatenate([train_y_origin, test_y_origin], axis=0), dtype='int')

# load FS data
repre_list = ['Avalon', 'ECFP', 'FCFP', 'Rdkit']
s = repre_list[1] + '_RFE'
load_data_path = './data/save_data_BM_FS/ECFP_RFE/'
train_x = np.load(os.path.join(load_data_path, s + '_' + 'train_x7.npy'))
train_y = np.load(os.path.join(load_data_path, s + '_' + 'train_y7.npy'))
test_x = np.load(os.path.join(load_data_path, s + '_' + 'test_x7.npy'))
test_y = np.load(os.path.join(load_data_path, s + '_' + 'test_y7.npy'))
x_data = np.array(np.concatenate([train_x, test_x], axis=0), dtype='int')
y_data = np.array(np.concatenate([train_y, test_y], axis=0), dtype='int')

# load FS feature
FS_path = './data/FS_RFE/ECFP_RFE_index.npy'
FS_index = np.load(FS_path)
print('RFE model select %d features' % (len(FS_index)))

# # shap model
X_summary = shap.kmeans(x_data, 20)  # shap train data, rapid speed
# 将train_x 和shap_value转换为pd形式
x_data_df = pd.DataFrame(x_data, columns=[str(int(i)) for i in list(FS_index)])  # 原特征

# (1)save_path---SVM_model
save_data_path = './data/Shap_SVM_RFE_Results/'  # 变换路径
os.makedirs(save_data_path, exist_ok=True)

# load model
SVM_model = joblib.load('./data/FS_BM_SVM_Results/ECFP_RFE/FS_BM_SVM_ECFP_RFE7.pkl')  # 变换路径

explainer_SVM = shap.KernelExplainer(SVM_model.predict, X_summary, random_state=0)
print('Shap SVM model construct')
joblib.dump(explainer_SVM, os.path.join(save_data_path, 'shap_SVM_model.pkl'))
shap_values_SVM = explainer_SVM.shap_values(x_data)  # shap interpreter x_data

# plot two importance figure
fig, ax1 = plt.subplots()
shap.summary_plot(shap_values=shap_values_SVM,
                  features=x_data,
                  feature_names=x_data_df.columns,
                  show=False,
                  max_display=Top_N)
plt.title('SVM', fontsize=24)
plt.xticks(fontproperties='Times New Roman', size=24)  # 设置x坐标字体和大小
plt.yticks(fontproperties='Times New Roman', size=24)  # 设置y坐标字体和大小
plt.xlabel('Shap value(impact on model output)', fontsize=24)  # 设置x轴标签和大小
plt.tight_layout()  # 让坐标充分显示，如果没有这一行，坐标可能显示不全
fig.savefig(os.path.join(save_data_path, 'Feature_influence_save%d.png' % Top_N), dpi=1000)  # 可以保存图片

MA_shap_value = np.mean(abs(shap_values_SVM), axis=0)
list_shap = list(MA_shap_value)  # 计算每个特征的平均shap value
sorted_shap = sorted(MA_shap_value, reverse=True)  # 排序
select_top_N = sorted_shap[0:Top_N]  # 取排序前10的特征
select_index_list = []
for top_value in select_top_N:
    index = list_shap.index(top_value)  # 对应FS的index
    origin_feature = FS_index[index]
    select_index_list.append(origin_feature)  # 对应原特征 排序前TopN的特征
np.save(os.path.join(save_data_path, s + '_' + 'shap_Top_%d_index.npy' % Top_N), select_index_list)

workbook = xlwt.Workbook()
sheet = workbook.add_sheet('Exp_results')
sheet.write(0, 1, 'Overall')
sheet.write(0, 2, 'Non_tox')
sheet.write(0, 3, 'Tox')
sheet.write(0, 4, 'The ratio of Tox/Non')
sheet.write(0, 5, 'The ratio of Tox')
sheet.write(0, 6, 'Overall')
sheet.write(0, 7, 'Non_tox')
sheet.write(0, 8, 'Tox')
sheet.write(0, 9, 'The ratio of Tox/Non')
sheet.write(0, 10, 'The ratio of Tox')

# function：包含子结构的label分布图
for i in range(len(select_index_list)):
    feature_index = select_index_list[i]
    print('Top%d_feature_%d' % (i, feature_index))
    x_data_index = x_data_origin[:, feature_index]

    print('Satis exist feature results------------')
    exist_feature_list = np.where(x_data_index == 1)  # 存在该特征
    exist_corres_y = y_data_origin[exist_feature_list]  # 存在该特征， 对应的label
    Non_tox_num = np.size((np.where(exist_corres_y == 1)))
    Tox_num = np.size(np.where(exist_corres_y == 0))
    if Non_tox_num != 0:
        ratio_tox_Non = Tox_num / Non_tox_num
    else:
        ratio_tox_Non = 0
    ratio_Tox = Tox_num / len(exist_corres_y)

    sheet.write(i + 1, 0, 'Top%d_feature_%d' % (i, feature_index))
    sheet.write(i + 1, 1, '%d' % (len(exist_corres_y)))
    sheet.write(i + 1, 2, '%d' % (Non_tox_num))
    sheet.write(i + 1, 3, '%d' % (Tox_num))
    sheet.write(i + 1, 4, '%.2f' % (ratio_tox_Non))
    sheet.write(i + 1, 5, '%.2f%%' % (ratio_Tox * 100))

    print('Satis Non exist feature results-----------')
    Non_exist_feature_list = np.where(x_data_index == 0)  # 存在该特征
    Non_exist_corres_y = y_data_origin[Non_exist_feature_list]  # 存在该特征， 对应的label
    Non_tox_num = np.size((np.where(Non_exist_corres_y == 1)))
    Tox_num = np.size(np.where(Non_exist_corres_y == 0))
    if Non_tox_num != 0:
        ratio_tox_Non = Tox_num / Non_tox_num
    else:
        Non_tox_num = 0
    ratio_Tox = Tox_num / len(Non_exist_corres_y)

    sheet.write(i + 1, 6, '%d' % (len(Non_exist_corres_y)))
    sheet.write(i + 1, 7, '%d' % (Non_tox_num))
    sheet.write(i + 1, 8, '%d' % (Tox_num))
    sheet.write(i + 1, 9, '%.2f' % (ratio_tox_Non))
    sheet.write(i + 1, 10, '%.2f%%' % (ratio_Tox * 100))
workbook.save(os.path.join(save_data_path, 'Shap_SVM_Results_%s.xls' % (s)))

# (2)save_path---RF_model
save_data_path = './data/Shap_RF_RFE_Results/'  # 变换路径
os.makedirs(save_data_path, exist_ok=True)

# load model
RF_model = joblib.load('./data/FS_BM_RF_Results/ECFP_RFE/FS_BM_RF_ECFP_RFE7.pkl')  # 变换路径

# shap model
# X_summary = shap.kmeans(x_data, 10)  # shap train data, rapid speed
explainer_RF = shap.KernelExplainer(RF_model.predict, X_summary, random_state=0)
print('Shap RF model construct')
joblib.dump(explainer_RF, os.path.join(save_data_path, 'shap_RF_model.pkl'))
shap_values_RF = explainer_RF.shap_values(x_data)  # shap interpreter x_data

# plot two importance figure
fig, ax1 = plt.subplots()
shap.summary_plot(shap_values=shap_values_RF,
                  features=x_data,
                  feature_names=x_data_df.columns,
                  show=False,
                  max_display=Top_N)
plt.title('RF', fontsize=24)
plt.xticks(fontproperties='Times New Roman', size=24)  # 设置x坐标字体和大小
plt.yticks(fontproperties='Times New Roman', size=24)  # 设置y坐标字体和大小
plt.xlabel('Shap value(impact on model output)', fontsize=24)  # 设置x轴标签和大小
plt.tight_layout()  # 让坐标充分显示，如果没有这一行，坐标可能显示不全
fig.savefig(os.path.join(save_data_path, 'Feature_influence_save%d.png' % Top_N), dpi=1000)  # 可以保存图片

MA_shap_value = np.mean(abs(shap_values_RF), axis=0)
list_shap = list(MA_shap_value)  # 计算每个特征的平均shap value
sorted_shap = sorted(MA_shap_value, reverse=True)  # 排序
select_top_N = sorted_shap[0:Top_N]  # 取排序前10的特征
select_index_list = []
for top_value in select_top_N:
    index = list_shap.index(top_value)  # 对应FS的index
    origin_feature = FS_index[index]
    select_index_list.append(origin_feature)  # 对应原特征 排序前TopN的特征
np.save(os.path.join(save_data_path, s + '_' + 'shap_Top_%d_index.npy' % Top_N), select_index_list)

workbook = xlwt.Workbook()
sheet = workbook.add_sheet('Exp_results')
sheet.write(0, 1, 'Overall')
sheet.write(0, 2, 'Non_tox')
sheet.write(0, 3, 'Tox')
sheet.write(0, 4, 'The ratio of Tox/Non')
sheet.write(0, 5, 'The ratio of Tox')
sheet.write(0, 6, 'Overall')
sheet.write(0, 7, 'Non_tox')
sheet.write(0, 8, 'Tox')
sheet.write(0, 9, 'The ratio of Tox/Non')
sheet.write(0, 10, 'The ratio of Tox')

# function：包含子结构的label分布图
for i in range(len(select_index_list)):
    feature_index = select_index_list[i]
    print('Top%d_feature_%d' % (i, feature_index))
    x_data_index = x_data_origin[:, feature_index]

    print('Satis exist feature results------------')
    exist_feature_list = np.where(x_data_index == 1)  # 存在该特征
    exist_corres_y = y_data_origin[exist_feature_list]  # 存在该特征， 对应的label
    Non_tox_num = np.size((np.where(exist_corres_y == 1)))
    Tox_num = np.size(np.where(exist_corres_y == 0))
    if Non_tox_num != 0:
        ratio_tox_Non = Tox_num / Non_tox_num
    ratio_Tox = Tox_num / len(exist_corres_y)

    sheet.write(i + 1, 0, 'Top%d_feature_%d' % (i, feature_index))
    sheet.write(i + 1, 1, '%d' % (len(exist_corres_y)))
    sheet.write(i + 1, 2, '%d' % (Non_tox_num))
    sheet.write(i + 1, 3, '%d' % (Tox_num))
    sheet.write(i + 1, 4, '%.2f' % (ratio_tox_Non))
    sheet.write(i + 1, 5, '%.2f%%' % (ratio_Tox * 100))

    print('Satis Non exist feature results-----------')
    Non_exist_feature_list = np.where(x_data_index == 0)  # 存在该特征
    Non_exist_corres_y = y_data_origin[Non_exist_feature_list]  # 存在该特征， 对应的label
    Non_tox_num = np.size((np.where(Non_exist_corres_y == 1)))
    Tox_num = np.size(np.where(Non_exist_corres_y == 0))
    if Non_tox_num != 0:
        ratio_tox_Non = Tox_num / Non_tox_num
    ratio_Tox = Tox_num / len(Non_exist_corres_y)
    sheet.write(i + 1, 6, '%d' % (len(Non_exist_corres_y)))
    sheet.write(i + 1, 7, '%d' % (Non_tox_num))
    sheet.write(i + 1, 8, '%d' % (Tox_num))
    sheet.write(i + 1, 9, '%.2f' % (ratio_tox_Non))
    sheet.write(i + 1, 10, '%.2f%%' % (ratio_Tox * 100))
workbook.save(os.path.join(save_data_path, 'Shap_RF_Results_%s.xls' % (s)))

# (3)save_path---GBDT_model
save_data_path = './data/Shap_GBDT_RFE_Results/'  # 变换路径
os.makedirs(save_data_path, exist_ok=True)

# load model
GBDT_model = joblib.load('./data/FS_BM_GBDT_Results/ECFP_RFE/FS_BM_GBDT_ECFP_RFE7.pkl')  # 变换路径

# shap model
# X_summary = shap.kmeans(x_data, 10)  # shap train data, rapid speed
explainer_GBDT = shap.KernelExplainer(GBDT_model.predict, X_summary, random_state=0)
print('Shap GBDT model construct')
joblib.dump(explainer_GBDT, os.path.join(save_data_path, 'shap_GBDT_model.pkl'))
shap_values_GBDT = explainer_GBDT.shap_values(x_data)  # shap interpreter x_data

# plot two importance figure
fig, ax1 = plt.subplots()
shap.summary_plot(shap_values=shap_values_GBDT,
                  features=x_data,
                  feature_names=x_data_df.columns,
                  show=False,
                  max_display=Top_N)
plt.title('GBDT', fontsize=24)
plt.xticks(fontproperties='Times New Roman', size=24)  # 设置x坐标字体和大小
plt.yticks(fontproperties='Times New Roman', size=24)  # 设置y坐标字体和大小
plt.xlabel('Shap value(impact on model output)', fontsize=24)  # 设置x轴标签和大小
plt.tight_layout()  # 让坐标充分显示，如果没有这一行，坐标可能显示不全
fig.savefig(os.path.join(save_data_path, 'Feature_influence_save%d.png' % Top_N), dpi=1000)  # 可以保存图片

MA_shap_value = np.mean(abs(shap_values_GBDT), axis=0)
list_shap = list(MA_shap_value)  # 计算每个特征的平均shap value
sorted_shap = sorted(MA_shap_value, reverse=True)  # 排序
select_top_N = sorted_shap[0:Top_N]  # 取排序前10的特征
select_index_list = []
for top_value in select_top_N:
    index = list_shap.index(top_value)  # 对应FS的index
    origin_feature = FS_index[index]
    select_index_list.append(origin_feature)  # 对应原特征 排序前TopN的特征
np.save(os.path.join(save_data_path, s + '_' + 'shap_Top_%d_index.npy' % Top_N), select_index_list)

workbook = xlwt.Workbook()
sheet = workbook.add_sheet('Exp_results')
sheet.write(0, 1, 'Overall')
sheet.write(0, 2, 'Non_tox')
sheet.write(0, 3, 'Tox')
sheet.write(0, 4, 'The ratio of Tox/Non')
sheet.write(0, 5, 'The ratio of Tox')
sheet.write(0, 6, 'Overall')
sheet.write(0, 7, 'Non_tox')
sheet.write(0, 8, 'Tox')
sheet.write(0, 9, 'The ratio of Tox/Non')
sheet.write(0, 10, 'The ratio of Tox')

# function：包含子结构的label分布图
for i in range(len(select_index_list)):
    feature_index = select_index_list[i]
    print('Top%d_feature_%d' % (i, feature_index))
    x_data_index = x_data_origin[:, feature_index]

    print('Satis exist feature results------------')
    exist_feature_list = np.where(x_data_index == 1)  # 存在该特征
    exist_corres_y = y_data_origin[exist_feature_list]  # 存在该特征， 对应的label
    Non_tox_num = np.size((np.where(exist_corres_y == 1)))
    Tox_num = np.size(np.where(exist_corres_y == 0))
    if Non_tox_num != 0:
        ratio_tox_Non = Tox_num / Non_tox_num
    ratio_Tox = Tox_num / len(exist_corres_y)

    sheet.write(i + 1, 0, 'Top%d_feature_%d' % (i, feature_index))
    sheet.write(i + 1, 1, '%d' % (len(exist_corres_y)))
    sheet.write(i + 1, 2, '%d' % (Non_tox_num))
    sheet.write(i + 1, 3, '%d' % (Tox_num))
    sheet.write(i + 1, 4, '%.2f' % (ratio_tox_Non))
    sheet.write(i + 1, 5, '%.2f%%' % (ratio_Tox * 100))

    print('Satis Non exist feature results-----------')
    Non_exist_feature_list = np.where(x_data_index == 0)  # 不存在该特征
    Non_exist_corres_y = y_data_origin[Non_exist_feature_list]  # 不存在该特征， 对应的label
    Non_tox_num = np.size((np.where(Non_exist_corres_y == 1)))
    Tox_num = np.size(np.where(Non_exist_corres_y == 0))
    if Non_tox_num != 0:
        ratio_tox_Non = Tox_num / Non_tox_num
    ratio_Tox = Tox_num / len(Non_exist_corres_y)

    sheet.write(i + 1, 6, '%d' % (len(Non_exist_corres_y)))
    sheet.write(i + 1, 7, '%d' % (Non_tox_num))
    sheet.write(i + 1, 8, '%d' % (Tox_num))
    sheet.write(i + 1, 9, '%.2f' % (ratio_tox_Non))
    sheet.write(i + 1, 10, '%.2f%%' % (ratio_Tox * 100))

workbook.save(os.path.join(save_data_path, 'Shap_GBDT_Results_%s.xls' % (s)))
