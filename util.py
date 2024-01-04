from sklearn import metrics
import pandas as pd

def get_adult_income_data():
    data_train = pd.read_csv('./data/adult/train.csv') 
    data_test = pd.read_csv('./data/adult/test.csv')

    data_train = data_train.to_numpy()
    data_test = data_test.to_numpy()

    x_train = data_train[:, :-1]
    y_train = data_train[:, -1]

    x_test = data_test[:, :-1]
    y_test = data_test[:, -1]

    return x_train, y_train, x_test, y_test

def get_adult_income_features():
    data_train = pd.read_csv('./data/adult/train.csv') 

    cols = list(data_train.columns)
    feature_names = cols[:-1]
    label_name = cols[-1]
    return feature_names, label_name

def get_adult_income_feature_category_mapping():
    feature_mapping = {
        'age': ['age'], 
        'capital_gain': ['capital_gain'], 
        'capital_loss': ['capital_loss'], 
        'hours_per_week': ['hours_per_week'], 
        'workclass': ['workclass_Federal_gov', 'workclass_Local_gov', 'workclass_Never_worked', 
                      'workclass_Private', 'workclass_Self_emp_inc', 'workclass_Self_emp_not_inc', 
                      'workclass_State_gov', 'workclass_Without_pay'], 
        'education': ['education_Bachelors', 'education_Doctorate', 'education_HS_grad', 
                      'education_Masters', 'education_Prof_school', 'education_School', 
                      'education_Some_college', 'education_Voc'], 
        'marital_status': ['marital_status_divorced', 'marital_status_married', 'marital_status_not_married'], 
        'occupation': ['occupation_Adm_clerical', 'occupation_Armed_Forces', 'occupation_Craft_repair', 
                       'occupation_Exec_managerial', 'occupation_Farming_fishing', 'occupation_Handlers_cleaners', 
                       'occupation_Machine_op_inspct', 'occupation_Other_service', 'occupation_Priv_house_serv', 
                       'occupation_Prof_specialty', 'occupation_Protective_serv', 'occupation_Sales', 'occupation_Tech_support', 
                       'occupation_Transport_moving'], 
        'relationship': ['relationship_Husband', 'relationship_Not_in_family', 'relationship_Other_relative', 
                         'relationship_Own_child', 'relationship_Unmarried', 'relationship_Wife'], 
        'race': ['race_Amer_Indian_Eskimo', 'race_Asian_Pac_Islander', 'race_Black', 'race_Other', 'race_White'], 
        'sex': ['sex_Female', 'sex_Male'], 
        'native_country': ['native_country_Other', 'native_country_United_States']
    }
    return feature_mapping
    
def get_model_auc(model, x, y):
    y_pred = model.predict_proba(x)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
    auc = metrics.auc(fpr, tpr)
    return auc

def get_rule_len(rule, feature_to_category_mapping):
    features = [f.strip() for f in rule.split("&")]
    features = [feature_to_category_mapping[f.split()[0]] for f in features]
    features = list(set(features))
    return len(features)

def get_feature_to_category_mapping(category_to_feature_mapping):
    feature_to_category_mapping = {}
    for k, v in category_to_feature_mapping.items():
        for i in range(len(v)):
            feature_to_category_mapping[v[i]] = k
    return feature_to_category_mapping    
    
def get_most_interpretable_rules(rules_for_data, all_rules_from_model, category_to_feature_mapping):
    feature_to_category_mapping = get_feature_to_category_mapping(category_to_feature_mapping)
    rule_len = {}
    for r in all_rules_from_model:
        rule_len[r] = get_rule_len(r, feature_to_category_mapping)

    MAX_RULE_LEN = len(list(feature_to_category_mapping.keys()))
    
    shortest_rule_for_data = []
    shortest_rule_len_for_data = []
    for i in range(len(rules_for_data)):
        shortest_rule_for_data.append([])
        shortest_rule_len_for_data.append(MAX_RULE_LEN)
        if(len(rules_for_data[i]) > 0):
            min_rule_len  = rule_len[rules_for_data[i][0]]
            min_rule  = rules_for_data[i][0]
            for j in range(len(rules_for_data[i])):
                if(min_rule_len > rule_len[rules_for_data[i][j]]):
                    min_rule_len  = rule_len[rules_for_data[i][j]]
                    min_rule  = rules_for_data[i][j]
        shortest_rule_for_data[i] = [min_rule]
        shortest_rule_len_for_data[i] = min_rule_len
    
    return shortest_rule_for_data

def get_category_mappings_from_data(x, features, category_to_feature_map):
    feature_to_category_map = get_feature_to_category_mapping(category_to_feature_map)
    
    x_feature_map = {} 
    for i in range(len(x)):
        x_feature_map[features[i]] = x[i]

    x_category_map = {}
    for k in x_feature_map.keys():
        category = feature_to_category_map[k]
        x_category_map[category] = []
        for fi in category_to_feature_map[category]:
            x_category_map[category].append(x_feature_map[fi])
    return x_category_map

def display_input(x, features, category_to_feature_map):
    x_category_map = get_category_mappings_from_data(x, features, category_to_feature_map)
    print("Input data with", len(x_category_map), "features:")        
    for k, v in x_category_map.items():
        if(len(v) == 1):
            value = v[0]
            print(k, ":", value)
        else:
            for i in range(len(v)):
                if(v[i] == 1):
                    value = category_to_feature_map[k][i]
                    value = value[len(k) + 1:]
                    print(k, ":", value)
    return

def get_counterfactual_explanation(x, feature_names, features_that_cannot_be_changed, all_rules_from_model):
    df = pd.DataFrame(x, columns=feature_names)
        
    possible_counterfactuals = set()
    for r in all_rules_from_model:
        terms = r.split("&")
        terms = [t.strip() for t in terms]
        counterfactual = []
        for t in terms:
            support = df.query(str(t)).index.tolist()
            if(len(support) == 0):
                counterfactual.append(t)
        counterfactual_str = " & ".join(counterfactual)
        for f in features_that_cannot_be_changed:
            if(f in counterfactual_str):
                counterfactual = []
                counterfactual_str = ""
        if((len(counterfactual) == 1) and (counterfactual_str not in possible_counterfactuals)):
            possible_counterfactuals.add(counterfactual_str)
    possible_counterfactuals = list(possible_counterfactuals)
    
    return simplify_counterfactuals(possible_counterfactuals)

def simplify_counterfactuals(counterfactuals):       
    counterfactuals_feature = [f.split(" ")[0].strip() for f in counterfactuals]
    counterfactuals_sign = [f.split(" ")[1].strip() for f in counterfactuals]
    counterfactuals_value = [f.split(" ")[2] for f in counterfactuals]
    
    counterfactuals_feature_map = {}
    for i in range(len(counterfactuals_feature)):
        k = counterfactuals_feature[i] + " " + counterfactuals_sign[i]
        v = counterfactuals_value[i]
        if(k not in counterfactuals_feature_map):
            counterfactuals_feature_map[k] = []
        counterfactuals_feature_map[k] = counterfactuals_feature_map[k] + [v]
    
    for k in counterfactuals_feature_map.keys():
        sign = k.split(" ")[1]
        if(sign == ">"):
            counterfactuals_feature_map[k] = min(counterfactuals_feature_map[k])
        elif(sign == "<="):
            counterfactuals_feature_map[k] = max(counterfactuals_feature_map[k])
        else:
            raise ValueError("Unknown sign")
    
    simplified_counterfactuals = []
    for k, v in counterfactuals_feature_map.items():
        simplified_counterfactuals.append(k + " " + v)
    return simplified_counterfactuals
    