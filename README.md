# news-stock-predictor
Predicts stock movements using deep neural networks to classify news headline events

Utilizes the PyTorch AI library to create Deep Neural Network models to extract important event embeddings in news headlines.
Once event embeddings are extracted, events are aligned in groups (month, week, day) and passed through a convolutional
neural network to filter the most important events and decide if the overall news weighting for the time period is positive or negative.
