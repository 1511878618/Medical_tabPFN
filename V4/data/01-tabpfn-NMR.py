import pandas as pd
import json
from config import *
from ppp_prediction.utils import load_data
from ppp_prediction.model import fit_best_model
import pickle


import matplotlib.pyplot as plt
import seaborn as sns
from ppp_prediction.plot.utils import save_fig

params = {
    "axes.labelsize": 14,  # fontsize for x and y labels (was 10)
    "font.size": 8,  # was 10
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": False,
    "figure.figsize": [5, 5],
    "font.family": "Calibri",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 2,
    # set title fontsize
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "figure.titlesize": 15,
    # label
    "axes.labelweight": "bold",
}

plt.rcParams.update(params)

diseaserawDir = dataDir / "label"



def get_predict_v2_from_df(
    model,
    data,
    x_var,
):
    """
    merge by idx
    """

    no_na_data = data[x_var].dropna().copy()
    if hasattr(model, "predict_proba"):
        no_na_data["pred"] = model.predict_proba(no_na_data)[:, 1]
    else:
        no_na_data["pred"] = model.predict(no_na_data)

    return (
        data[[]]
        .merge(no_na_data[["pred"]], left_index=True, right_index=True, how="left")
        .values.flatten()
    )

NMR_train = load_data(dataDir / "imputeData" / "NMR_train.feather")
held_out_train = load_data(dataDir / "imputeData" / "held_out_train.feather")
held_out_test = load_data(dataDir / "imputeData" / "held_out_test.feather")
print(NMR_train.info())




def get_predict_v2_from_df(
    model,
    data,
    x_var,
):
    """
    merge by idx
    """

    no_na_data = data[x_var].dropna().copy()
    if hasattr(model, "predict_proba"):
        no_na_data["pred"] = model.predict_proba(no_na_data)[:, 1]
    else:
        no_na_data["pred"] = model.predict(no_na_data)

    return (
        data[[]]
        .merge(no_na_data[["pred"]], left_index=True, right_index=True, how="left")
        .values.flatten()
    )



drop_dict = {
    "T2D": ["Glucose", "glucose", "glycated_haemoglobin_hba1c"],
    "Stroke": None,
    "Heart_Failure": None,
    "Coronary_Artery_Disease": None,
    "Chronic_kidney_disease": ["Creatinine", "creatinine"],
    "Atrial_fibrillation_or_flutter": None,
    "PAD": None,
    "Hypertension": None,
    "Venous_Thromboembolism": None,
    "CMD": ["Creatinine", "creatinine", "Glucose", "glucose"],
}


def drop_value_from_list(l, v=None):
    if v is None:
        return l
    if len(v) == 0:
        return l
    will_drop = []
    will_keep = []
    for i in l:
        if i in v:
            will_drop.append(i)
        else:
            will_keep.append(i)
    print(f"will drop {will_drop}")
    return will_keep
    # will_drop.extend(drop_dict[i])

list(diseaserawDir.glob("*"))

from ppp_prediction.model_v2.models import (
    fit_best_model_v2,
    fit_ensemble_model_simple_v2,
    fit_lightgbm,
    fit_xgboost,
)

modelSaveDir = outputDir / "0x-tabpfn-NMR"


import numpy as np 
def get_sig_vars(tgt_dir = "V2/output/02-risk-model/02-Result/", tgt_name="NMR", suffix = "glmnet/bootstrap/bootstrap_coef_df.csv", topk=50):
    tgt_dir = Path(tgt_dir)
    all_disease = list(tgt_dir.glob("*"))

    res_dict = {}
    for disease in all_disease:
        disease_name = disease.name
        tgt_coef_file = tgt_dir / disease_name / tgt_name/ suffix
        if tgt_coef_file.exists():
            coef_df = pd.read_csv(tgt_coef_file)
            coef_df.set_index(coef_df.columns[0], inplace=True)
            coef_df = np.abs(coef_df.mean(axis=1)).sort_values(ascending=False)
            coef_df = coef_df[coef_df > 0].head(topk).index.tolist()

            res_dict[disease_name] = coef_df
    return res_dict
            
            
            
            
    

from collections import defaultdict
from ppp_prediction.cox import run_cox

# from ppp_prediction.model import run_glmnet
E = "incident"
T = "survTime"
disease_dict = defaultdict(dict)

for disease_dir in diseaserawDir.glob("*"):
    disease_name = disease_dir.stem


    disease_df = load_data(disease_dir)

    disease_train = disease_df.merge(NMR_train, on="eid", how="inner")
    disease_data_held_out_train = disease_df.merge(
        held_out_train, on="eid", how="inner"
    )
    disease_data_held_out_test = disease_df.merge(held_out_test, on="eid", how="inner")

    disease_modelSaveDir = modelSaveDir / disease_name
    disease_modelSaveDir.mkdir(parents=True, exist_ok=True)
    if (disease_modelSaveDir / "metrics.csv").exists():
        print(f"{disease_name} already exists")
        
        # c_disease_dict = pickle.load(
        #     open(disease_modelSaveDir / "disease_dict.pkl", "rb")
        # )
    else:
        final_need_cols = ["eid", "incident", "survTime", "date"]
        c_model_dict = {}

        # for name, combination in clinical_risk_dict.items():
        # name = "PANEL"
        # combination = clinical_risk_dict[name]
        combination = list(NMR_train.columns)[1:]
        

        # split data 

        # 1) full data ; 2) random sample data 
        n_case_train = int(disease_train["incident"].sum())
    
        disease_train = disease_train.dropna(subset=["incident", *combination])
        disease_train_case = disease_train.query("incident == 1").sample(n=5000 if n_case_train > 5000 else n_case_train)
        # disease_train_control = disease_train.query("incident == 0").sample(n=5000 if n_case_train > 5000 else n_case_train)
        disease_train_control = disease_train.query("incident == 0").sample(n=5000)


        # train model 
        # metrics_res = [] 
        score_name_list = ["lasso_full", "xgboost_full"]
        # basic_cols = ["eid", E, T, "incident", "survTime", "date"]
        
        disease_data_held_out_train = disease_data_held_out_train.dropna(
            subset=["incident", *combination]
        )
        X_held_out_train = disease_data_held_out_train[combination]
        y_held_out_train = disease_data_held_out_train["incident"]

        disease_data_held_out_test = disease_data_held_out_test.dropna(
            subset=["incident", *combination]
        )
        X_held_out_test = disease_data_held_out_test[combination]
        y_held_out_test = disease_data_held_out_test["incident"]
         


        # Test model 1) full lasso xgboost 2) sampled and features from 5, 10, 20, 50, 100, 200 of lasso, AutoTabPFN, xgboost

        # 1) Lasso full  
        lasso_full_savedir = disease_modelSaveDir / "lasso_full.pkl"
        if lasso_full_savedir.exists():
            lasso_full = pickle.load(open(lasso_full_savedir, "rb"))
            print(f"{disease_name} lasso_full loaded")
 
        else: 
            (lasso_full, *_) = fit_best_model(
                train_df=disease_train,
                test_df=disease_data_held_out_test,
                X_var=combination,
                y_var="incident",
                method_list="Lasso",
                cv=3,
                engine="cuml",
            )
                
            pickle.dump(
                lasso_full, open(lasso_full_savedir, "wb")
            )
        
        disease_data_held_out_test["lasso_full"] = get_predict_v2_from_df(
            lasso_full, disease_data_held_out_test, combination
        )

        del lasso_full

        # xgboost full 
        xgboot_full_savedir = disease_modelSaveDir / "xgboost_full.pkl"
        if (disease_modelSaveDir / "xgboost_full.pkl").exists():
            print(f"{disease_name} xgboost_full loaded")
            xgboost_full_tuned = pickle.load(open(xgboot_full_savedir, "rb"))
        else:
            
            xgboost_full_tuned, *_ = fit_xgboost(
                train = disease_train,
                xvar = combination,
                label = "incident",
                tuning = True,
                tune_config = {"max_iter": 100}
            )
            pickle.dump(
                xgboost_full_tuned, open(xgboot_full_savedir, "wb")
            )
        disease_data_held_out_test["xgboost_full"] = get_predict_v2_from_df(
            xgboost_full_tuned, disease_data_held_out_test, combination
        )

        del xgboost_full_tuned

        
        for topk in [5, 10, 20, 50, 100 ,200]:
            print(f"{disease_name} {topk}")
            combination_dict = get_sig_vars(tgt_name = "NMR", topk=topk)
            sig_combination = combination_dict[disease_name]
            disease_train_sample = pd.concat([disease_train_case, disease_train_control])[['eid', E, T ] + sig_combination]
            X_train = disease_train_sample[sig_combination]
            y_train = disease_train_sample["incident"]
            
            lasso_sample_topk_savedir = disease_modelSaveDir / f"lasso_sample_{topk}.pkl"
            if lasso_sample_topk_savedir.exists():
                lasso_sample = pickle.load(open(lasso_sample_topk_savedir, "rb"))
                print(f"{disease_name} lasso_sample_{topk} loaded")
            else:
                try:
                    (lasso_sample, *_) = fit_best_model(
                        train_df=disease_train_sample,
                        test_df=disease_data_held_out_test,
                        X_var=sig_combination,
                        y_var="incident",
                        method_list="Lasso",
                        cv=3,
                        engine="cuml",
                    )
                    pickle.dump(
                    lasso_sample, open(disease_modelSaveDir / f"lasso_sample_{topk}.pkl", "wb")
                )
                except:
                    print(f"{disease_name} lasso_sample_{topk} failed")
                
            disease_data_held_out_test[f"lasso_sample_{topk}"] = get_predict_v2_from_df(
                            lasso_sample, disease_data_held_out_test, sig_combination
            )
 
            del lasso_sample
            # try:
            #     from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
                
            #     AutoTabPFN_topk_savedir = disease_modelSaveDir / f"AutoTabPFN_{topk}.pkl"
            #     if AutoTabPFN_topk_savedir.exists():
            #         AutoTabPFN = pickle.load(open(AutoTabPFN_topk_savedir, "rb"))
            #         print(f"{disease_name} AutoTabPFN_{topk} loaded")
            #     else:
    
            #             AutoTabPFN = AutoTabPFNClassifier(max_time=120, device='cuda') # 120 seconds tuning time
            #             AutoTabPFN.fit(X_train, y_train)
            #             pickle.dump(
            #             AutoTabPFN, open(disease_modelSaveDir / f"AutoTabPFN_{topk}.pkl", "wb")
            #         )
            #     # disease_data_held_out_test["AutoTabPFN"] = AutoTabPFN.predict_proba(X_held_out_test)[:, 1]
            #     disease_data_held_out_test[f"AutoTabPFN_{topk}"] = AutoTabPFN.predict_proba(disease_data_held_out_test[sig_combination])[:, 1]
        
            #     del AutoTabPFN
            # except:
            #     print(f"AutoTabPFN {topk} failed")

            try:
                from tabpfn import TabPFNClassifier
                
                TabPFN_topk_savedir = disease_modelSaveDir / f"TabPFN{topk}.pkl"
                if TabPFN_topk_savedir.exists():
                    TabPFN = pickle.load(open(TabPFN_topk_savedir, "rb"))
                    print(f"{disease_name} TabPFN_{topk} loaded")
                else:
    
                        TabPFN = TabPFNClassifier(device='cuda:0', ignore_pretraining_limits=True)
                        TabPFN.fit(X_train, y_train)
                        pickle.dump(
                        TabPFN, open(disease_modelSaveDir / f"TabPFN_{topk}.pkl", "wb")
                    )
                # disease_data_held_out_test["AutoTabPFN"] = AutoTabPFN.predict_proba(X_held_out_test)[:, 1]
                disease_data_held_out_test[f"TabPFN_{topk}"] = TabPFN.predict_proba(disease_data_held_out_test[sig_combination])[:, 1]
        
                del TabPFN
            except:
                print(f"TabPFN {topk} failed")

            # xgboost sampled
            xgboost_sample_savedir = disease_modelSaveDir / f"xgboost_sample_{topk}.pkl"
            if xgboost_sample_savedir.exists():
                xgboost_sample_tuned = pickle.load(open(xgboost_sample_savedir, "rb"))
                print(f"{disease_name} xgboost_sample_{topk} loaded")
            else:
                try:
                    xgboost_sample_tuned, *_ = fit_xgboost(
                    train = disease_train_sample,
                    xvar = sig_combination,
                    label= "incident",
                    tuning= True,
                    tune_config = {"max_iter": 100}
                )
                    pickle.dump(
                    xgboost_sample_tuned, open(disease_modelSaveDir / f"xgboost_sample_{topk}.pkl", "wb")
                )
                except Exception as e:
                    print(f"{disease_name} xgboost_sample_{topk} failed")
                    

            disease_data_held_out_test[f"xgboost_sample_{topk}"] = get_predict_v2_from_df(
                xgboost_sample_tuned, disease_data_held_out_test, sig_combination
            )

            del xgboost_sample_tuned
            score_name_list.extend([f"lasso_sample_{topk}", f"xgboost_sample_{topk}", f"TabPFN_{topk}"])
            


        held_out_test_df_to_save = disease_data_held_out_test[["eid", "incident", "survTime", "date"] + score_name_list].reset_index(drop=True)
        held_out_test_df_to_save.to_feather(disease_modelSaveDir / "held_out_test.feather")
        from ppp_prediction.metrics import cal_binary_metrics
        metrics_list = [] 
        for key in score_name_list:
            if key not in disease_data_held_out_test.columns:
                continue 
            to_cal_df = disease_data_held_out_test[["eid", E, T, key]].copy().dropna()
            res = cal_binary_metrics(to_cal_df["incident"], to_cal_df[key], n_resamples= 30, ci = True)
            # res = run_cox(to_cal_df, var=key, E=E, T=T, ci=True, n_resamples=100)
            res["disease"] = disease_name
            res["method"] = key
            metrics_list.append(res)
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(disease_modelSaveDir / "metrics.csv", index=False)
         
        