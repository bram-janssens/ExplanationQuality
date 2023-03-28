# ExplanationQuality

This repository contains the implementation code for a study on explanation quality. It contains implementations which can either use fitted priori learners (deployed on TML, RNNs, and CNNs), as well as an implementation where the evaluation is done within the folding (applied on transformer model). 

The methodology and results are currently under review at European Journal of Operational Research.

The main analysis are in Transformers.ipynb, LIME_Lean_Loop_Covid.ipynb and LIME_Lean_Loop_CovidDL.ipynb. The permutations per observation are created in PermutationScriptTML.ipynb and ParallelCovid.py to save computational time, but can be conducted within the main analyses.

Practitioners interested in using the methodology on other data sets should focus on the following for loop:

```
for i in range(train.shape[0]):
  true_obs = train.iloc[i,:]
  key = true_obs['Unnamed: 0']
  if key!=1629:
      permutations_obs = save_dict[key]
      tussen = permutations_obs.drop(columns = 'id')
      interpretable_x = tussen.filter(regex='^\\D')
      x_lime = tussen[tussen.columns.drop(list(tussen.filter(regex='bow_')))],
      x_lime = x_lime.drop(columns = x_lime.filter(regex='\\.').columns)
      x_lime = x_lime.drop(columns = 'text')
      y_lime = model.predict_proba(x_lime)[:, 1]
      true_obs = true_obs[x_lime.columns
      kernel_values = pairwise_kernels(true_obs.values.reshape(1, -1), x_lime, metric='linear')[0]
      if min(kernel_values)<0:
          kernel_values = kernel_values-(min(kernel_values)-1)
      interpretable_x = interpretable_x.drop(columns = 'text')
      used_features = feat_selection(interpretable_x, y_lime, kernel_values, 20)
      uitkomst = explain_instance_with_data(interpretable_x, y_lime, kernel_values, used_features)
      uitkomsten.append(uitkomst)
```

The individual explanations are stored in the uitkomsten object, which can be evaluated with the syntax used afterwards.

As it is infeasible to permute empty texts (e.g., only contain visual keys) in a meaningful way (i.e., all permutations represent the same empty texts, generating no variability in x_lime and x_interpretable), these should be omitted or handled explicitely, as is done for observation 1629 in the COVID data set.
