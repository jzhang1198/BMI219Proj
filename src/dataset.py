import pandas as pd
import numpy as np
import torch
import os

class SeiDataset(torch.utils.data.Dataset):
    "Custom dataset class"

    def __init__(self, df, feature_list):
        self.observations = df[feature_list].to_numpy()
        self.labels = df['is_associated'].to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        observation = self.observations[index]
        label = self.labels[index]
        return observation, label 

def load_data():
    data_dir = '/Users/jonathanzhang/Documents/ucsf/bmi219/data/sei_data'
    control_df = None
    gwas_df = None
    for i in range (1,11):

        if type(control_df) != pd.DataFrame:
            control_df = pd.read_csv(os.path.join(data_dir, f"random_snps_{i}.tsv"),sep='\t')
            gwas_df = pd.read_csv(os.path.join(data_dir, f"SeiResults_{i}.tsv"),sep='\t')

        else:
            control_df = pd.concat([control_df, pd.read_csv(os.path.join(data_dir, f"random_snps_{i}.tsv"),sep='\t')])
            gwas_df = pd.concat([gwas_df, pd.read_csv(os.path.join(data_dir, f"SeiResults_{i}.tsv"),sep='\t')])

    gwas_df = gwas_df[gwas_df['ref_match']==True]
    gwas_df = gwas_df.assign(is_associated = 1)
    control_df = control_df[control_df['ref_match']==True]
    control_df = control_df.assign(is_associated = 0)

    feature_list = list(control_df.columns[9:-1])
    label_col = 'is_associated'

    df = pd.concat([gwas_df, control_df])
    df = df.reset_index(drop = True)

    return df, feature_list

def split_data(df, feature_list: list, batch_size: int, length_list=[159740, 19968, 19968]):

    #split data into train, test, and validation sets
    train_split, val_split, test_split = torch.utils.data.random_split(df, lengths=length_list)
    train_df = train_split.dataset.iloc[train_split.indices]
    val_df = val_split.dataset.iloc[val_split.indices]
    test_df = test_split.dataset.iloc[test_split.indices]

    train_dataloader = torch.utils.data.DataLoader(SeiDataset(train_df,feature_list), batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(SeiDataset(val_df,feature_list), batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(SeiDataset(test_df,feature_list), batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
