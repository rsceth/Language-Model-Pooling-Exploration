from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


def dataBatcher(tokenizer, train_dataset, test_dataset):
    """
        tokenize, padding and batch dataset
        ============================================
        args : tokenizer <transformer tokenizer>, 
                train_dataset, test_dataset  <transformer dataset format>
        return : train_dataloader, eval_dataloader <transformer dataloader>
    """
    def preprocess_function(examples):
        """ tokenizer to meet datframe format"""
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True)
    tokenized_train_datasets = tokenized_train_datasets.remove_columns(['text'])

    tokenized_test_datasets = test_dataset.map(preprocess_function, batched=True)
    tokenized_test_datasets = tokenized_test_datasets.remove_columns(['text'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_train_datasets, shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_test_datasets, batch_size=8, collate_fn=data_collator
    )
    return train_dataloader, eval_dataloader