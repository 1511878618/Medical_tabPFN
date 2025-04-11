from sklearn.model_selection import train_test_split
from ppp_prediction.metrics import cal_binary_metrics


# define a function to fit the model, save the result and collect the scores
## parallel this function
def fit_model_and_save_result(
    total_df,  # first col should be eid
    label_df,  # first col should be eid
    save_dir,
    # asssoc_df=None, # used to sort and get top features to downsample, if None then will do assoc
    feature_rank_list=None,  # used to downsample the features
    min_class_number_cutoff={"train": 30, "validation": 10, "test": 10},
    train_test_split_ratio=0.7,
    seed=1234,
    device="cuda",
    topk_list=[5, 10, 20, 50, 100],
):
    """
    1. merge the total_df and disease_df
    2. check label_df sum is over min_class_number_cutoff; if not return None, and print the error
    3. fit the models: 1) full lasso, 2) full xgboost, 3) sample lasso, 4) sample xgboost, 5) sample AutoTabPFN, 6) TabPFN
    4. save the results: 1) model, 2) scores of total_df, 3) metrics of models
    """
    # step1: merge and check

    ## check the first column
    if total_df.columns[0] != "eid":
        raise ValueError("total_df first column should be eid")
    if label_df.columns[0] != "eid":
        raise ValueError("label_df first column should be eid")

    features = total_df.columns[1:].tolist()
    label = label_df.columns[1]

    ## merge
    merged_df = pd.merge(total_df, label_df, on="eid", how="inner")
    print(
        f"Found merged samples: {merged_df.shape[0]} while, label_df: {label_df.shape[0]} and total_df: {total_df.shape[0]}"
    )

    # step2: fit the model
    ## step2.1 Train Test Split

    save_dir.mkdir(parents=True, exist_ok=True)
    train_test_split_feather_dir = save_dir / f"train_test_split_total.feather"
    
    if train_test_split_feather_dir.exists():
        train_test_split_eid = pd.read_feather(train_test_split_feather_dir)
        
    else:
        train_df_eid, test_df_eid = train_test_split(
            merged_df[['eid']], test_size=1 - train_test_split_ratio, random_state=seed
        )
        train_test_split_eid = pd.concat(
            [
                train_df_eid[["eid"]].copy().assign(Type="train"),
                test_df_eid[["eid"]].copy().assign(Type="test"),
            ]
        ).reset_index(drop=True)
        train_test_split_eid.to_feather(train_test_split_feather_dir)
    train_df = pd.merge(train_test_split_eid.query("Type == 'train'"), merged_df, on="eid", how="inner")
    test_df = pd.merge(train_test_split_eid.query("Type == 'test'"), merged_df, on="eid", how="inner")
    print(f"Train Test Split Done: {train_df.shape[0]}  {test_df.shape[0]}")

    merged_df = merged_df.merge(train_test_split_eid, on="eid", how="inner")
    
    scores = []  # save the scores
    ## step2.2 check the min_class_number
    for min_class_number_check_key in ["train", "test"]:
        if min_class_number_check_key == "train":
            to_check_df = train_df
        elif min_class_number_check_key == "test":
            to_check_df = test_df
        else:
            raise ValueError(
                "min_class_number_check_key should be in ['train', 'validation', 'test']"
            )
        min_class_number = min(to_check_df[label].value_counts())
        # the min class number and class name
        min_class_name = to_check_df[label].value_counts().idxmin()
        if min_class_number < min_class_number_cutoff[min_class_number_check_key]:
            print(
                f"Error: {min_class_number_check_key} {min_class_name} has only {min_class_number} samples, less than {min_class_number_cutoff[min_class_number_check_key]}"
            )
            return None
    print(
        f"Train data have {train_df.shape[0]} samples with {train_df[label].sum():.0f} cases"
    )

    print(
        f"Test data have {test_df.shape[0]} samples with {test_df[label].sum():.0f} cases"
    )

    ## check
    train_meta_info = {}
    ## step2.4 fit the models
    modelSaveDir = save_dir / "models"
    modelSaveDir.mkdir(parents=True, exist_ok=True)
    ### 1) Lasso full


    lasso_full_savedir = modelSaveDir / "lasso_full.pkl"
    if lasso_full_savedir.exists():
        lasso_full = pickle.load(open(lasso_full_savedir, "rb"))
        # print(f"lasso_full loaded")

    else:
        lasso_engine = "cuml" if device == "cuda" else "sklearn"
        print(merged_df.shape)
        if merged_df.shape[0] < 5000:
            lasso_engine = "sklearn"
        print(f"lasso_full start with engine {lasso_engine}")

        (lasso_full, *_) = fit_best_model(
            train_df=train_df,
            X_var=features,
            y_var=label,
            method_list="Lasso",
            cv=5,
            engine=lasso_engine,
        )

        pickle.dump(lasso_full, open(lasso_full_savedir, "wb"))

    merged_df["lasso_full"] = get_predict_v2_from_df(lasso_full, merged_df, features)
    scores.append("lasso_full")
    
    # return lasso_full, score_df
    train_meta_info[f"lasso_full"] = {
        "train_case": train_df[label].sum(),
        "train_control": train_df.shape[0] - train_df[label].sum(),
    }
    if isinstance(feature_rank_list, str):
        if feature_rank_list == "assoc":
            # feature_rank_list = lasso_full.coef_.argsort()
            raise NotImplementedError("assoc not implemented")
            # pass
        elif feature_rank_list == "lasso":
            feature_rank_df = pd.DataFrame(
                [lasso_full.feature_names_in_, lasso_full[-1].coef_]
            ).T
            feature_rank_df.columns = ["feature", "coef"]
            feature_rank_df["abs_coef"] = feature_rank_df["coef"].abs()
            feature_rank_df = feature_rank_df.sort_values("abs_coef", ascending=False)

            feature_rank_df.to_csv(
                save_dir / "feature_rank_lasso_full.csv", index=False
            )
            feature_rank_list = feature_rank_df.query("abs_coef != 0 ")["feature"].tolist()
        else:
            raise ValueError(
                "feature_rank_list should be in ['assoc', 'lasso'] or a list of features with the first one is the most important"
            )
    elif isinstance(feature_rank_list, list):
        pass
    else:
        raise ValueError(
            "feature_rank_list should be in ['assoc', 'lasso'] or a list of features with the first one is the most important"
        )

    del lasso_full
    
    # xgboost full
    xgboot_full_savedir = modelSaveDir / "xgboost_full.pkl"
    if (modelSaveDir / "xgboost_full.pkl").exists():
        print(f"xgboost_full loaded")
        xgboost_full_tuned = pickle.load(open(xgboot_full_savedir, "rb"))
    else:

        xgboost_full_tuned, *_ = fit_xgboost(
            train=train_df,
            xvar=features,
            label=label,
            tuning=True,
            tune_config={"max_iter": 100},
        )
        pickle.dump(xgboost_full_tuned, open(xgboot_full_savedir, "wb"))
    merged_df["xgboost_full"] = get_predict_v2_from_df(
        xgboost_full_tuned, merged_df, features
    )
    scores.append("xgboost_full")

    train_meta_info[f"xgboost_full"] = {
        "train_case": train_df[label].sum(),
        "train_control": train_df.shape[0] - train_df[label].sum(),
    }

    del xgboost_full_tuned

    for strata in [
        "balance", 
                   "random"
                  ]:  # balance or random
        strata_train_test_split_feather_dir =  save_dir / f"train_test_split_{strata}.feather"

            
        if strata_train_test_split_feather_dir.exists():
            disease_train_sample_eid = pd.read_feather(strata_train_test_split_feather_dir)
            print(f"Strata {strata} Train Test Split Loading from disk: {disease_train_sample_eid.shape[0]}")
        else:
            if strata == "balance":
                disease_train_case = train_df.query(f"{label} == 1")
                disease_train_case_number = min(disease_train_case.shape[0], 5000)
    
                disease_train_case = disease_train_case.sample(
                    n = disease_train_case_number , random_state=seed, replace=False
                )
                
                disease_train_control = train_df.query(f"{label} == 0").sample(
                    n=disease_train_case.shape[0], random_state=seed
                )
                disease_train_sample_eid = pd.concat(
                    [disease_train_case, disease_train_control]
                )[['eid']]
            elif strata == "random":
                disease_train_sample_eid = train_df.sample(n=10000, random_state=seed)[['eid']] if train_df.shape[0] > 10000 else train_df[['eid']]
            disease_train_sample_eid.reset_index(drop=True).to_feather(strata_train_test_split_feather_dir)
            print(f"Strata {strata} Train Test Split Done: {disease_train_sample_eid.shape[0]}")
        disease_train_sample = pd.merge(disease_train_sample_eid, train_df, on="eid", how="inner")
        
        for topk in topk_list:
            sig_features = feature_rank_list[:topk]

            suffix_name = f"{topk}_{strata}"

            print(suffix_name)

            strata_topk_save_dir = modelSaveDir / f"{topk}/{strata}"
            strata_topk_save_dir.mkdir(parents=True, exist_ok=True)

            X_train = disease_train_sample[sig_features]
            y_train = disease_train_sample[label]

            print("Lasso Start")
            lasso_sample_topk_savedir = strata_topk_save_dir / f"lasso_sample.pkl"
            if lasso_sample_topk_savedir.exists():
                lasso_sample = pickle.load(open(lasso_sample_topk_savedir, "rb"))
                print(f"lasso_sample_{suffix_name} loaded")
            else:
                try:
                    lasso_engine = "cuml" if device == "cuda" else "sklearn"
                    if X_train.shape[0] < 5000:
                        lasso_engine = "sklearn"
                    print(f"lasso_full start with engine {lasso_engine}")

                    (lasso_sample, *_) = fit_best_model(
                        train_df=disease_train_sample,
                        X_var=sig_features,
                        y_var=label,
                        method_list="Lasso",
                        cv=5,
                        engine=lasso_engine,
                    )
                    pickle.dump(
                        lasso_sample,
                        open(strata_topk_save_dir / f"lasso_sample.pkl", "wb"),
                    )
                except Exception as e:
                    print(f"lasso_sample_{topk} failed and erros: {e}")

            merged_df[f"lasso_sample_{suffix_name}"] = get_predict_v2_from_df(
                lasso_sample, merged_df, sig_features
            )
            scores.append(f"lasso_sample_{suffix_name}")
                
            train_meta_info[f"lasso_sample_{suffix_name}"] = {
                "train_case": disease_train_sample[label].sum(),
                "train_control": disease_train_sample.shape[0]
                - disease_train_sample[label].sum(),
            }

            del lasso_sample
            print("autoTapFPN")
            from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
                AutoTabPFNClassifier,
            )

            AutoTabPFN_topk_savedir = strata_topk_save_dir / f"AutoTabPFN.pkl"
            if AutoTabPFN_topk_savedir.exists():
                AutoTabPFN = pickle.load(open(AutoTabPFN_topk_savedir, "rb"))
                print(f"AutoTabPFN_{topk} loaded")
            else:

                try:
                    AutoTabPFN = AutoTabPFNClassifier(
                        max_time=40, device="cuda" if device == "cuda" else "cpu", ignore_pretraining_limits=True
                    )  # 120 seconds tuning time
                    AutoTabPFN.fit(X_train, y_train)
                    pickle.dump(
                        AutoTabPFN, open(strata_topk_save_dir / f"AutoTabPFN.pkl", "wb")
                    )
                except Exception as e:
                    print(f"AutoTabPFN_{topk} failed with {e}")

            # score_df["AutoTabPFN"] = AutoTabPFN.predict_proba(X_held_out_test)[:, 1]
            merged_df[f"AutoTabPFN_{suffix_name}"] = AutoTabPFN.predict_proba(
                merged_df[sig_features]
            )[:, 1]
            scores.append(f"AutoTabPFN_{suffix_name}")
            train_meta_info[f"AutoTabPFN_{suffix_name}"] = {
                "train_case": disease_train_sample[label].sum(),
                "train_control": disease_train_sample.shape[0]
                - disease_train_sample[label].sum(),
            }

            del AutoTabPFN
            print(f"tabpfn")
            try:
                from tabpfn import TabPFNClassifier

                TabPFN_topk_savedir = strata_topk_save_dir / f"TabPFN{topk}.pkl"
                if TabPFN_topk_savedir.exists():
                    TabPFN = pickle.load(open(TabPFN_topk_savedir, "rb"))
                    
                    print(f"TabPFN_{topk} loaded")
                else:

                    TabPFN = TabPFNClassifier(
                        device="cuda:0" if device == "cuda" else "cpu",
                        ignore_pretraining_limits=True,
                    )
                    TabPFN.fit(X_train, y_train)
                    pickle.dump(
                        TabPFN, open(strata_topk_save_dir / f"TabPFN.pkl", "wb")
                    )
                    # score_df["AutoTabPFN"] = AutoTabPFN.predict_proba(X_held_out_test)[:, 1]
                    merged_df[f"TabPFN_{suffix_name}"] = TabPFN.predict_proba(
                        merged_df[sig_features]
                    )[:, 1]
                    scores.append(f"TabPFN_{suffix_name}")
                train_meta_info[f"TabPFN_{suffix_name}"] = {
                    "train_case": disease_train_sample[label].sum(),
                    "train_control": disease_train_sample.shape[0]
                    - disease_train_sample[label].sum(),
                }
                del TabPFN
            except Exception as e:
                print(f"TabPFN {topk} failed with {e}")

            # xgboost sampled
            print(f"xgboost now")
            xgboost_sample_savedir = strata_topk_save_dir / f"xgboost_sample.pkl"
            if xgboost_sample_savedir.exists():
                xgboost_sample_tuned = pickle.load(open(xgboost_sample_savedir, "rb"))
                print(f"xgboost_sample_{topk} loaded")
            else:
                try:
                    xgboost_sample_tuned, *_ = fit_xgboost(
                        train=disease_train_sample,
                        xvar=sig_features,
                        label=label,
                        tuning=True,
                        tune_config={"max_iter": 100},
                    )
                    pickle.dump(
                        xgboost_sample_tuned,
                        open(strata_topk_save_dir / f"xgboost_sample.pkl", "wb"),
                    )
                except Exception as e:
                    print(f"xgboost_sample_{topk} failed")

            merged_df[f"xgboost_sample_{suffix_name}"] = get_predict_v2_from_df(
                xgboost_sample_tuned, merged_df, sig_features
            )
            scores.append(f"xgboost_sample_{suffix_name}")
            train_meta_info[f"xgboost_sample_{suffix_name}"] = {
                "train_case": disease_train_sample[label].sum(),
                "train_control": disease_train_sample.shape[0]
                - disease_train_sample[label].sum(),
            }

            del xgboost_sample_tuned
    score_df = merged_df[["eid", label, "Type"] + scores]
    score_df.to_feather(save_dir / "score.feather")
    pickle.dump(train_meta_info, open(save_dir / "train_meta_info.pkl", "wb"))
    metrics_list = []
    for key in scores:  # eid label Type
        to_cal_df = (
            score_df.query("Type == 'test'")[["eid", label, key]].copy().dropna()
        )
        res = cal_binary_metrics(
            to_cal_df[label], to_cal_df[key], n_resamples=30, ci=True
        )
        # res = run_cox(to_cal_df, var=key, E=E, T=T, ci=True, n_resamples=100)
        res["method"] = key
        res.update(train_meta_info[key])
        metrics_list.append(res)
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(save_dir / "metrics.csv", index=False)

from ppp_prediction.model_v2.models import (
    fit_best_model_v2,
    fit_ensemble_model_simple_v2,
    fit_lightgbm,
    fit_xgboost,
)


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


if __name__ == "__main__":

    from config import *
    import pandas as pd
    import json
    from ppp_prediction.utils import load_data
    from ppp_prediction.model import fit_best_model
    import pickle
    from tqdm import tqdm
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    from ppp_prediction.plot.utils import save_fig
    
    # Define Basic Variables
    
    groupByVar = "Ethnic"  # Ethnic
    omicsName = "Prot_meanImpute"  # used omics Name
    phenoDefineVersion = "fatal_disease"  # used pheno version
    
    ## cutoff
    Case_cutoff = 50  # only over this number of cases will be used as a phenotype
    
    # Define the dirs
    covariates_dir = dataDir / "covariates.feather"
    omicsDataDir = dataDir / f"Prot/{omicsName}.feather"
    
    phenoDefineDir = dataDir / f"{phenoDefineVersion}"
    
    # load data
    covariates_df = pd.read_feather(covariates_dir)
    omicsData = pd.read_feather(omicsDataDir)
    print(
        f"Total {omicsData.shape[0]} samples and {omicsData.shape[1]} features with {omicsName}"
    )
    diseaseList = list(phenoDefineDir.glob("*.feather"))
    foundedPhenoFile = len(list(phenoDefineDir.glob("*.feather")))
    print(f"Founded Pheno Files: {foundedPhenoFile}")
    covariates_df
    
    # groupByVar used
    used_groupByVar = ["White"]
    covariates_df[groupByVar].value_counts()
    
    # show all disease rate in Prot
    
    
    res_dict = {
        "event": [],
        "incident": [],
        "prevalent": [],
    }
    for disease in tqdm(diseaseList, total=len(diseaseList), desc="Counting..."):
        df = pd.read_feather(disease).query("eid in @omicsData.eid")
    
        for col in ["event", "incident", "prevalent"]:
            case = int(df[col].sum())
            control = int(df.shape[0] - case)
            rate = case / df.shape[0]
            res_dict[col].append(
                pd.DataFrame(
                    {
                        "Phenotype": [disease.stem],
                        "Case": [case],
                        "Control": [control],
                        "Rate": [rate],
                    }
                )
            )
    
    
    
    event_df = (
        pd.concat(res_dict["event"])
        .sort_values("Rate", ascending=False)
        .reset_index(drop=True)
    )
    incident_df = (
        pd.concat(res_dict["incident"])
        .sort_values("Rate", ascending=False)
        .reset_index(drop=True)
    )
    prevalent_df = (
        pd.concat(res_dict["prevalent"])
        .sort_values("Rate", ascending=False)
        .reset_index(drop=True)
    )
    
    event_df.to_csv(outputDir/"event_df.csv")
    prevalent_df.to_csv(outputDir/"prevalent_df.csv")
    incident_df.to_csv(outputDir/"incident_df.csv")
    
    
    # resultDir = outputDir / "Lancet_Digital_Health_2019"
    resultDir = outputDir / phenoDefineVersion
    
    # first run White
    for disease in diseaseList:
        diseaseName = disease.stem
        for c_groupbyVar in used_groupByVar:
            c_groupbyVar_eids = covariates_df.query(f"{groupByVar} == @c_groupbyVar").eid
            for label in ["incident", "prevalent"]:
                c_save_dir  = resultDir / f"{diseaseName}/{c_groupbyVar}/{label}"
                print(f"Currently, Running {label} {diseaseName} and {c_groupbyVar}")
                if (c_save_dir / "metrics.csv").exists():
                    print(f"Currently, {label} {diseaseName} and {c_groupbyVar} exists.")
                    continue
                label_df = pd.read_feather(disease).query("eid in @omicsData.eid")
                res = fit_model_and_save_result(
                    total_df=omicsData.query(f"eid in @c_groupbyVar_eids"),
                    label_df=label_df[["eid", label]],
                    save_dir=c_save_dir,
                    feature_rank_list="lasso",  # currently use the fake rank list, better use assoc on whole set or run assoc on the train set
                    device="cuda",
                    # topk_list=[5, 10, 20, 50, 100, 200],
                    topk_list = [5, 10, 20]
                )
   