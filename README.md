# LLM Project
Alexandra Snelling

## Project Task
Task: Document Summarization 
The project aims to fine-tune an LLM capable of summarization on a set of news articles. 

## Dataset
Dataset: 'multi_news'
This dataset contains contains 44972 entries of full news stories, listed as 'document', each with a corresponding human-generated 'summary'. 

## Pre-trained Model
Model: T5-Small
This is a text-to-text LLM created by Google. This means that the model can both take in text as an input and generate text as an output. 
The "small" variant of T5 attempts to maximize resource efficiency, while maintaining a relatively high performance.
Being resource efficient is important for machines with limited computational resources.
The T5-Small model is capable of multiple tasks so the "summarize" prefix is used when calling it, in order to acheive the desired output.

## Pre-processing
The data was preprocessed using the T5-Small associated Tokenizer.
(AutoTokenizer.from_pretrained("google-t5/t5-small"))
Use of the associated tokenizer is required for all LLM models in order to ensure that data in preprocessed in the same manner as the data that the model was trained on.

## Performance Metrics
Rouge scores are employed here as performance metrics. 
These are metrics specifically used for evaluating text summarization and machine translation quality. 

Rouge1: Measures single words in both the generated summary and the given summary.
Rouge2: Measures pairs of consecutive words in both the generated summary and the given summary.
RougeL: Measures the longest common sequence in both the generated summary and the given summary. 

For all Rouge scores, the higher the score, the more overlap the generated summary and the given summary.
This typically indicates more desireable results.

## Optimization and Hyperparameter Tuning

**Data Amount:** The first 'tuning' (although not a hyper-parameter) was to experiment with increasing training data amount. 
I was unable to run the entire dataset with the limited computing power available but I wanted to see the effect of increasing training data amount.
I experimented by using 5x more training documents (800 vs 4000 training documents). Results were as follows:

![model training using 800 training documents]("multi_news_train_800_t5-small_summary_model_training_metrics.jpg")

![model training using 4000 training documents]("multi_news_train_4000_t5-small_summary_model_training_metrics.jpg")

We see that a larger training set improved the Rouge 1 and Rouge L scores. 
We can also see that all Rouge scores improve with each epoch in the larger training set. 

**Note:** Due to limited GPU time availability on Google Colab, most hyperparameter tuning was not possible in the permitted time.
(Notebook would crash while using CPU and I ran out of Colab GPU time twice! I am unable to complete any more tuning at this time.)



Given more computing power, I would attempt the following hyperparameter tuning:


**Epochs:** Increase number of training epochs from 3 to 5 to allow more training iteration 
This can be especially helpful for summarization as it is a very complex task.

num_epochs = 5


**Learning Rate:** I would use a learning rate scheduler and assign warmup steps to adjust the learning rate dynamically while training.
This can help the model to avoid getting stuck in local gradient minima while training, while still speeding up the model optimization.

lr_scheduler_type="linear"
warmup_steps=X

(warmup steps are usually set to 10-20% of total training steps calculated as:
total_training_steps = (num_train_examples / per_device_train_batch_size) * num_epochs)


**LLM Model:** I would attempt a different LLM model that is also capable of summrization such as DistilBART with the DistilBART tokenizer.

## Deployment
The model was deployed and can be found at the following url:

https://huggingface.co/AlexandraSnelling/multi_news_train_4000_t5-small_summary_model/commit/3b18c76c913028eed574bdf4db168bc35af5f69a

![screenshot of model on HuggingFace]("huggingface_model_screenshot.jpg")