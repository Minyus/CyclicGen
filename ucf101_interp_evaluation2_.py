import pandas as pd

from pathlib import Path, PureWindowsPath, WindowsPath

# with Path('./ucf101_interp_ours/ucf101_interp_evaluation.csv').open() as f:
#     eval_df = pd.read_csv(f, index_col='model_id')
eval_df = pd.read_csv('ucf101_interp_ours\\ucf101_interp_evaluation.csv', index_col='index')

print(eval_df.head(5))

# with Path('./train_dir/model_param_lookup.csv').open() as f:
#     lu_df = pd.read_csv(f, index_col='model_id')
lu_df = pd.read_csv('train_dir\\model_param_lookup.csv', index_col='model_id')
print(lu_df.head(5))
out_df = pd.merge(eval_df, lu_df, how='left', left_on='model_id', right_index=True)
print(out_df.head(5))
with Path('./ucf101_interp_ours/ucf101_interp_evaluation_params.csv').open('w') as f:
    out_df.to_csv(f, line_terminator='\n')
