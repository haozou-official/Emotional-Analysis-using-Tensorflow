# Emotional-Analysis-using-Tensorflow
LSTM in emotional analysis of IMDB Film critics(using Tensorflow)<br>
Data Sources:Total 20000 movie reviews in IMDB from Kaggle dataset<br>
Positive evaluation:10000 reviews<br>
Negative evaluation:10000 reviews<br>
The projects iterates for 100,000 times<br>
training for nearly 4 hours<br>
the correct rate gradually converge, nearly 3/4 of the time to reach 100% correct rate<br>
* The initial goal was to call the model after successful training to achieve the application of the emotional judgment of the corpus, but it was later found that there were some problems with this idea, and it was necessary to think of other applications cause simply by using the trained model to classify the emotion of the corpus and not achieve the role of prediction.<br>

# Project Comments
In the course of the project, I gradually found that many materials can be tried again and make the project better: <br>
* 1.With keras training rather than TensorFlow will be more efficient and faster, the former is equivalent to a streamlined version of the TensorFlow, and the speed will have a certain increase.
The process of TensorFlow model call is a bit complicated, and the training is really slow<br>
* 2.With the Naive Bayesian algorithm rather than LSTM. Simply with the LSTM model training words will not be much better, we need to enhance the LSTM model for some innovation and improvement.
But the choice of Naive Bayesian will actually be better, cause the essence of emotional analysis is a classification problem, and naïve Bayes is the most efficient in classification problems.

# Main Process
![wordList](https://github.com/zouhao0418/Emotional-Analysis-using-Tensorflow/blob/master/wordList.png)
![wordVectors](https://github.com/zouhao0418/Emotional-Analysis-using-Tensorflow/blob/master/wordVectors.png)<br>
## Check the index of the word ***children***<br>
![childrenIndex](https://github.com/zouhao0418/Emotional-Analysis-using-Tensorflow/blob/master/childrenIndex.png)<br>
## Define the correct predictive function and correct rate evaluation parameters<br>
```
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
```
## Finally, the standard cross entropy loss function is defined as the loss value, here is the ADAM as the optimization function<br>
```
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))                           
optimizer = tf.train.AdamOptimizer().minimize(loss)
```
## Call Tensorflow
```
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

```
## Iterates for 100,000 times
```
iterations = 100000
for i in range(iterations):
    # Data for the next batch
    next_batch, next_batch_labels = get_train_batch()
sess.run(optimizer,{input_data: next_batch, labels: next_batch_labels})
```
## Results
![training results](https://github.com/zouhao0418/Emotional-Analysis-using-Tensorflow/blob/master/training%20results.png)
