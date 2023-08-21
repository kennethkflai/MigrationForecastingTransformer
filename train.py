from datasets import load_dataset, load_from_disk
from functools import partial
import numpy as np
from tools import transform_start_field

from gluonts.time_feature import get_lags_for_frequency
from gluonts.time_feature import time_features_from_frequency_str

from tools import create_train_dataloader, create_test_dataloader
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from transformers import AutoformerConfig, AutoformerForPrediction
from transformers import InformerConfig, InformerForPrediction

import tqdm 


            
dataset = load_from_disk("data/")

# dataset2 = load_dataset("monash_tsf", "tourism_monthly")
# train2 = dataset2["train"]
# 
freq = "1M"
prediction_length = 12

train_dataset = dataset["train"]
test_dataset = dataset["test"]

train_dataset.set_transform(partial(transform_start_field, freq=freq))
test_dataset.set_transform(partial(transform_start_field, freq=freq))

lags_sequence = get_lags_for_frequency(freq)
time_features = time_features_from_frequency_str(freq)
d_model_dim = 64
def settings(model_type, index):
    if model_type == 0:
        config = AutoformerConfig(
            prediction_length=prediction_length,
            # context length:
            context_length=prediction_length * index,
            # lags coming from helper given the freq:
            lags_sequence=lags_sequence,
            # we'll add 2 time features ("month of year" and "age", see further):
            num_time_features=len(time_features)+1,
            # we have a single static categorical feature, namely time series ID:
            num_static_categorical_features=1,
            # it has 366 possible values:
            cardinality=[len(train_dataset)],
            # the model will learn an embedding of size 2 for each of the 366 possible values:
            embedding_dimension=[2],
            
            num_parallel_samples= 100,
            # transformer params:
            encoder_layers=4,
            decoder_layers=4,
            d_model=d_model_dim,
        )
            
        model = AutoformerForPrediction(config)
        
    elif model_type == 1:
        config = TimeSeriesTransformerConfig(
            prediction_length=prediction_length,
            # context length:
            context_length=prediction_length * index,
            # lags coming from helper given the freq:
            lags_sequence=lags_sequence,
            # we'll add 2 time features ("month of year" and "age", see further):
            num_time_features=len(time_features)+1,
            # we have a single static categorical feature, namely time series ID:
            num_static_categorical_features=1,
            # it has 366 possible values:
            cardinality=[len(train_dataset)],
            # the model will learn an embedding of size 2 for each of the 366 possible values:
            embedding_dimension=[2],
            
            num_parallel_samples= 100,
            # transformer params:
            encoder_layers=4,
            decoder_layers=4,
            d_model=d_model_dim,
        )
        
        model = TimeSeriesTransformerForPrediction(config)
    elif model_type == 2:
        config = InformerConfig(
            prediction_length=prediction_length,
            # context length:
            context_length=prediction_length * index,
            # lags coming from helper given the freq:
            lags_sequence=lags_sequence,
            # we'll add 2 time features ("month of year" and "age", see further):
            num_time_features=len(time_features)+1,
            # we have a single static categorical feature, namely time series ID:
            num_static_categorical_features=1,
            # it has 366 possible values:
            cardinality=[len(train_dataset)],
            # the model will learn an embedding of size 2 for each of the 366 possible values:
            embedding_dimension=[2],
            
            num_parallel_samples= 100,
            # transformer params:
            encoder_layers=4,
            decoder_layers=4,
            d_model=d_model_dim,
        )
        
        model = InformerForPrediction(config)   
        
    return model, config

for model_type in range(0,3):
    for index in range(1,10):
        
        model, config = settings(model_type, index)
        
        model.config.distribution_output
        
        train_dataloader = create_train_dataloader(
            config=config,
            freq=freq,
            data=train_dataset,
            batch_size=256,
            num_batches_per_epoch=1000,
        )
        
        test_dataloader = create_test_dataloader(
            config=config,
            freq=freq,
            data=test_dataset,
            batch_size=64,
        )
        
        # batch = next(iter(train_dataloader))
        # for k, v in batch.items():
        #     print(k, v.shape, v.type())
            
        from accelerate import Accelerator
        from torch.optim import AdamW
        
        accelerator = Accelerator()
        device = accelerator.device
        
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)
        
        model, optimizer, train_dataloader = accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
        )
        
        model.train()
        for epoch in tqdm.tqdm(range(100)):
            for batch in train_dataloader:
                optimizer.zero_grad()
                outputs = model(
                    static_categorical_features=batch["static_categorical_features"].to(device)
                    if config.num_static_categorical_features > 0
                    else None,
                    static_real_features=batch["static_real_features"].to(device)
                    if config.num_static_real_features > 0
                    else None,
                    past_time_features=batch["past_time_features"].to(device),
                    past_values=batch["past_values"].to(device),
                    future_time_features=batch["future_time_features"].to(device),
                    future_values=batch["future_values"].to(device),
                    past_observed_mask=batch["past_observed_mask"].to(device),
                    # future_observed_mask=batch["future_observed_mask"].to(device),
                )
                loss = outputs.loss
        
                # Backpropagation
                accelerator.backward(loss)
                optimizer.step()
        
            # print(epoch, loss.item())
                    
        model.eval()
        
        forecasts = []
        for batch in test_dataloader:
            outputs = model.generate(
                static_categorical_features=batch["static_categorical_features"].to(device)
                if config.num_static_categorical_features > 0
                else None,
                static_real_features=batch["static_real_features"].to(device)
                if config.num_static_real_features > 0
                else None,
                past_time_features=batch["past_time_features"].to(device),
                past_values=batch["past_values"].to(device),
                future_time_features=batch["future_time_features"].to(device),
                past_observed_mask=batch["past_observed_mask"].to(device),
            )
            forecasts.append(outputs.sequences.cpu().numpy())
            
        forecasts = np.vstack(forecasts)
        
        
        from evaluate import load
        from gluonts.time_feature import get_seasonality
        mase_metric = load("evaluate-metric/mase")
        smape_metric = load("evaluate-metric/smape")
        forecast_median = np.median(forecasts, 1)
        
        mase_metrics = []
        smape_metrics = []
        for item_id, ts in enumerate(test_dataset):
            training_data = ts["target"][:-prediction_length]
            ground_truth = ts["target"][-prediction_length:]
            mase = mase_metric.compute(
                predictions=forecast_median[item_id],
                references=np.array(ground_truth),
                training=np.array(training_data),
                periodicity=get_seasonality(freq),
            )
            mase_metrics.append(mase["mase"])
        
            smape = smape_metric.compute(
                predictions=forecast_median[item_id],
                references=np.array(ground_truth),
            )
            smape_metrics.append(smape["smape"])
            
        print(f"index: {index}, Type: {model_type}, MASE: {np.mean(mase_metrics)}, sMAPE: {np.mean(smape_metrics)}")
        
        f = open("result.txt","a")
        f.write(f"index: {index}, Type: {model_type}, MASE: {np.mean(mase_metrics)}, sMAPE: {np.mean(smape_metrics)}\n")
        f.close()
        
        # from tools import plot, calculate_loss
        # for i in range(13):
        #     plot(i, forecasts, prediction_length, test_dataset, freq)
        
        #     mse, mae = calculate_loss(i,forecasts,test_dataset,prediction_length)
            # print(i, mse, mae)