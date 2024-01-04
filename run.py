from util import get_adult_income_data, get_adult_income_features, get_adult_income_feature_category_mapping

x_train, y_train, x_test, y_test = get_adult_income_data()
feature_names, label_name = get_adult_income_features()
feature_mapping = get_adult_income_feature_category_mapping()

from sklearn.ensemble import GradientBoostingClassifier
print("Training XGBoost Model")
model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=1)
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

print()
print("Train Accuracy:", model.score(x_train, y_train))
print("Test Accuracy:", model.score(x_test, y_test))

from util import get_model_auc

print()
print("Train AUC:", get_model_auc(model, x_train, y_train))
print("Test AUC:", get_model_auc(model, x_test, y_test))

import te2rules
from te2rules.explainer import ModelExplainer
print()
print("Using TE2Rules version:", te2rules.__version__)

model_explainer = ModelExplainer(model=model, feature_names=feature_names)

num_train = int(0.1 * len(x_train))
rules = model_explainer.explain(X=x_train[:num_train], y=y_train_pred[:num_train], num_stages=2)
longer_rules = model_explainer.rule_builder.longer_rules

print()
print("Found", len(rules), "rules")
print("Found", len(longer_rules), "longer rules")

print()
print("Fidelity of explanations")
overall, positives, negatives = model_explainer.get_fidelity(X=x_train, y=y_train_pred)
print("Fidelity on positives in Train:", positives)
overall, positive, negative = model_explainer.get_fidelity(X=x_test, y=y_test_pred)
print("Fidelity on positives in Test:", positives)

print()
print("Global Explanations of the model")
for i in range(len(rules)):
    print("Rule", i, ":", rules[i])

x_positives = []
x_negatives = []
for i in range(len(y_test)):
    if(y_test_pred[i] == 1):
        x_positives.append(x_test[i])
    else:
        x_negatives.append(x_test[i])

from util import get_most_interpretable_rules, display_input
print()
print("Local Explanations of a particular model decision")

selected_rules = model_explainer.explain_instance_with_rules(x_positives, explore_all_rules = True)
selected_rules = get_most_interpretable_rules(selected_rules, model_explainer.longer_rules, feature_mapping)

print()
print("Explaining positive model prediction")
print()
display_input(x_positives[1], feature_names, feature_mapping)
print()
print("Prediction:", 1)
print()
print("Reason:")
explanation = selected_rules[1]
for i in range(len(explanation)):
    print("Rule", i+1, ":", explanation[i])

from util import get_counterfactual_explanation, display_input
features_that_cannot_be_changed = ['marital_status', 'relationship', 'race', 'sex', 'native_country', 'age']

counterfactuals = get_counterfactual_explanation(
                                        [x_negatives[1]], 
                                        feature_names, 
                                        features_that_cannot_be_changed,
                                        model_explainer.longer_rules)

print()
print("Counterfactual explanation for negative model prediction")
print()
display_input(x_negatives[1], feature_names, feature_mapping)
print()
print("Prediction:", 0)
print()
print("Counterfactual Explanations: Satisfying any one of these is enough for making the model give a positive label")
for i in range(len(counterfactuals)):
    print("Rule", i+1, ":", counterfactuals[i])


