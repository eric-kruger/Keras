# Keras Tensor Flow Neural Net for Processing the semantic analysis of tweets
# Need to double check my datasets are not mismatched (x_train and y_train etc)
# Able to acheive at best 0.70 accuracy

# Links
  # data set : http://help.sentiment140.com/for-students
  # Tutorial : https://tensorflow.rstudio.com/blog/text-classification-with-keras.html
  # More on Keras : https://blog.keras.io/
  # Clean tweets : https://stackoverflow.com/questions/43557254/how-to-clean-a-tweet-using-regex-without-removing-punctuations-and-hasthag
  # Clean tweets : https://stackoverflow.com/questions/31348453/how-do-i-clean-twitter-data-in-r
  # Tweet pre preprocessing: https://github.com/s/preprocessor
  # Info on LSTM models : http://colah.github.io/posts/2015-08-Understanding-LSTMs/
  # Transcribing from voice to text : https://www.alexkras.com/transcribing-audio-file-to-text-with-google-cloud-speech-api-and-python/

# Load the Libraries
library(keras)
library(stringr)
library(tidyverse)
library(magrittr)

# Custom Functions
  # Vectorize_sequences
  vectorize_sequences <- function(sequences, dimension = NULL) {
    # sequences <- t_train_lst
    # dimension <- nrow(xt_tt_word)
    # Creates an all-zero matrix of shape (length(sequences), dimension)
    results <- matrix(rep(0,dimension*length(sequences)),ncol=dimension)
    i <- 1
    for (i in 1:length(sequences)) {
      # Sets specific indices of results[i] to 1s
      results[i,unlist(sequences[[i]][1])] <- 1
    }
    results
  }

# Load the Data
  setwd("~/R/Keras")
  # t_testing  <- read_csv("DATA/testdata.manual.2009.06.14.csv", col_names = FALSE)
  # t_entire <- read_csv("DATA/training.1600000.processed.noemoticon.csv",col_names = FALSE)
    # save.image("tweet_data.Rdata")
    load("tweet_data.Rdata")
  c.n <- c("Polairity","ID","Date","Query","User","Tweet")
  colnames(t_testing) <- c.n; colnames(t_entire) <- c.n
  t_entire$Polairity[t_entire$Polairity==4] <- 1    # Recode the polarity to 1 

# Create a training data set
  n <- 1000 # size of train data per polarity 
  xtabs(~Polairity,data=t_entire)
  t_e1 <- t_entire %>% filter(Polairity==1)
  t_e0 <- t_entire %>% filter(Polairity==0)
  set.seed(1979)
  sdata1 <- sample(1:nrow(t_e1),n)
  sdata0 <- sample(1:nrow(t_e0),n)
  t_train1 <- t_e1[sdata1,]
  t_train0 <- t_e0[sdata0,]
  xtabs(~Polairity,data=t_train1)
  xtabs(~Polairity,data=t_train0)

  t_train <- bind_rows(t_train1,t_train0)
  xtabs(~Polairity,data=t_train)

  y_train <- t_train$Polairity

# Clean the tweets and convert them to dummy coded variables for each word
  t_train_word   <- lapply(t_train$Tweet,str_split, pattern="[[:punct:][:space:]]")
  t_train_word_v <- unlist(t_train_word)
  #t_train_word_v <- stringi::stri_trans_tolower(t_train_word_v)
  t_train_word_v %<>% as.factor()
  t_train_word_v %<>% as.numeric()
  xt_tt_word    <- as.data.frame(xtabs(~t_train_word_v)) %>% arrange(desc(Freq))
  xt_tt_word$N  <- 1:nrow(xt_tt_word)
  #xt_tt_word %<>% arrange(desc(Freq))
  #levels(t_train_word_v) <- levels(t_train_word_v)[xt_tt_word$N]
  
  
  t_train_lst <- relist(t_train_word_v,skeleton = t_train_word)
     t_train_word_f <- factor(t_train_word_v)


  # Convert to the data frame of binary variables for each word
  vs <- vectorize_sequences(t_train_lst,dimension=nrow(xt_tt_word)) 

# Build the model
  model <- keras_model_sequential() %>% 
    layer_dense(units = 16, activation = "relu", input_shape = c(nrow(xt_tt_word))) %>% 
    layer_dense(units = 16, activation = "relu") %>% 
    layer_dense(units = 1, activation = "sigmoid")

  model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )

# Create a validation data set from the orginial training dataset
  n2 <- 100 # Size of the validation data set
  set.seed(1982)
  val_indicies <- sample(1:nrow(vs),n2)
  x_val <- vs[val_indicies,]        # validation predictors
  x_train <- vs[-val_indicies,]     # train predictors
  
  y_val <- y_train[val_indicies]    # validation outcomes
  y_train <- y_train[-val_indicies] # train outcomes

# Run the model
  model %>% fit(x_train, y_train, epochs = 20, batch_size = 512)
  results <- model %>% evaluate(x_val, y_val) # Accuracy of validation steps
  results

# Extract the Validation Indicies for comparison 
  val <- t_train[val_indicies,]
  val$p <- model %>% predict(x_val) %>% round(2) # Prediction on the validation step




