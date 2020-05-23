################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

#remove temporary variables
rm(dl, ratings, movies, test_index, temp, movielens, removed)

library(dplyr)
library(caret)
library(ggplot2)

# preliminary checks on data
head(edx)
head(validation)
dim(edx)
dim(validation)
sum(is.na(edx))
sum(is.na(validation))

#Unique users and movies in the list
edx%>% summarize(unique_users=n_distinct(userId), unique_movies=n_distinct(movieId))

# Create a list of ratings with the number of ratings.
edx%>%group_by(rating)%>%summarize(n=n())%>%arrange(desc(n))

# Plot the ratings Vs no. of ratings
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()+
  ggtitle("Ratings Vs Number of ratings")+xlab("Each rating of movies")+
  ylab("Number of ratings")

# Get all move titles
movie_titles <- edx %>% select(movieId, title) %>% distinct()

# Plot the number of ratings per movie
edx%>% count(movieId) %>%
  ggplot(aes(n))+
  geom_histogram(bins = 30, color="black")+
  scale_x_log10()+
  ggtitle("Number of ratings for Movies")+xlab("Number of movies")+
  ylab("Number of ratings")

# Plot the number of ratings per user
edx%>% count(userId) %>%
  ggplot(aes(n))+
  geom_histogram(bins = 30, color="black")+
  scale_x_log10()+
  ggtitle("Number of ratings for Users")+xlab("Number of users")+
  ylab("Number of ratings")

###############################################
# Create training and test sets from edx set  #
###############################################

# Used 10% for test and 90% for train
set.seed(123, sample.kind="Rounding")
t_index <- createDataPartition(y = edx$rating, times = 1,
                               p = 0.1, list = FALSE)
train_set <- edx[-t_index,]
temp <- edx[t_index,]

# Keep movies and users that have a match in the train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

#remove temporary variables
rm(t_index, temp, removed)

#############################################
#  Data Modeling and Analysis               #
#############################################
# Define RMSE function to call it as needed
RMSE <- function(y, yhat){
  sqrt(mean((y - yhat)^2))
}

# simple guessing model
set.seed(123, sample.kind="Rounding")
guess_rating<-sample(c(0.5,1,1.5,2,2.5,3,3.5,4,4.5,5), length(test_set$rating), replace=TRUE)
# randomly selects a number between 0.5 and 5
guess_rmse<-RMSE(test_set$rating, guess_rating)

# create a summary to keep adding RMSE values for each model 
rmse_summary<-data.frame(method = "A guess model", RMSE = guess_rmse)

#Plot to show guess model predictions
hist(guess_rating, xlab="Ratings", ylab="Number of ratings", main="Histogram of the ratings in a Guess model")


# average model
mu_train<-mean(train_set$rating)
mu_rating<-rep(mu_train, length(test_set$rating))
mu_rmse<-RMSE(test_set$rating, mu_rating)
rmse_summary<-rbind(rmse_summary, data.frame(method = "An average model", RMSE = mu_rmse))

rmse_summary %>% knitr::kable()

# Movie effect model
# Some movies are generally rated higher, b_m bias due to movies
train_set%>%
  group_by(movieId) %>% 
  summarize(mov_avg = mean(rating))%>%
  ggplot(aes(mov_avg))+geom_histogram(color="black", bins=30)+
  ggtitle("Histogram of Movie ratings")+xlab("Average Rating")+ylab("Number of ratings")

# Top 10 rated movies based on avg by movie
train_set%>%
  group_by(movieId) %>% 
  summarize(title=first(title), count=n(), mov_avg = mean(rating))%>%
  arrange(desc(mov_avg))%>%
  slice(1:10)

# Bottom 10 rated movies based on avg by movie
train_set%>%
  group_by(movieId) %>% 
  summarize(title=first(title), count=n(), mov_avg = mean(rating))%>%
  arrange(mov_avg)%>%
  slice(1:10)

#Calculate the movie effect or bias
movie_bias <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_m = mean(rating - mu_train))

#Plot of the movie effect b_m
movie_bias %>% qplot(b_m, geom ="histogram", bins = 10, data = ., color = I("black"))
+ggtitle("Movie effect estimate")+xlab("b_m value")+
  ylab("Count of b_m values")

#predicting b_m for test_set
b_m_hat<-test_set%>%left_join(movie_bias, by='movieId')%>%.$b_m 

# rating prediction and rmse of movie effect
mov_bias_rating<-mu_train+b_m_hat
mov_bias_rmse<-RMSE(test_set$rating, mov_bias_rating)
rmse_summary<-rbind(rmse_summary, data.frame(method = "Movie Bias model", RMSE = mov_bias_rmse))
rmse_summary %>% knitr::kable()


# User effect model
# Some users generally rate higher, b_u bias due to users
train_set%>%
  group_by(userId) %>% 
  summarize(u_avg = mean(rating))%>%
  ggplot(aes(u_avg))+geom_histogram(color="black", bins=30)+
  ggtitle("Histogram of User ratings")+xlab("Average Rating")+ylab("Number of ratings")

# Users with top ratings based on avg by user
train_set%>%
  group_by(userId) %>% 
  summarize(count=n(), u_avg = mean(rating))%>%
  arrange(desc(u_avg))%>%
  slice(1:10)

# Bottom 10 rated movies based on avg by user
train_set%>%
  group_by(userId) %>% 
  summarize(count=n(), u_avg = mean(rating))%>%
  arrange(u_avg)%>%
  slice(1:10)

#Calculate the user effect or bias
user_bias <- train_set %>% 
  left_join(movie_bias, by='movieId')%>%
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu_train - b_m))

#Plot of the user effect b_u
user_bias %>% qplot(b_u, geom ="histogram", bins = 10, data = ., color = I("black"))
+ggtitle("User effect estimate")+xlab("b_u value")+
  ylab("Count of b_u values")

#predicting b_u for test_set
u_bias_rating<-test_set%>%
  left_join(movie_bias, by='movieId')%>%
  left_join(user_bias, by='userId')%>%
  mutate(b_u_hat=mu_train+b_m+b_u)%>%.$b_u_hat

# rating prediction and rmse of movie effect
u_bias_rmse<-RMSE(test_set$rating, u_bias_rating)
rmse_summary<-rbind(rmse_summary, data.frame(method = "Movie and User Bias model", RMSE = u_bias_rmse))
rmse_summary %>% knitr::kable()

# Observe residuals are very large, hence improvement was not great
test_set %>% left_join(movie_bias, by='movieId') %>%
  mutate(residual = rating - (mu_train + b_m)) %>%
  arrange(desc(abs(residual))) %>% 
  slice(1:10) %>%
  pull(title)

#List of movie names
movie_titles<-edx%>% select(movieId, title)%>% distinct()

#movies with 5 best and worst estimates of b_m along with the rating count
train_set%>%count(movieId)%>%left_join(movie_bias)%>%
  left_join(movie_titles, by='movieId')%>%
  arrange(desc(b_m))%>%
  select(title, b_m, n)%>% slice(1:5)
train_set%>%count(movieId)%>%left_join(movie_bias)%>%
  left_join(movie_titles, by='movieId')%>%
  arrange(b_m)%>%
  select(title, b_m, n)%>% slice(1:5)

# Regularization of movie effect
lambda<-3
movie_regn <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_reg_m = sum(rating - mu_train)/(n()+lambda), n_i = n()) 

#top 10 best movies based on b_m
train_set %>%
  count(movieId) %>% 
  left_join(movie_regn, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_reg_m)) %>% 
  slice(1:10) %>% 
  pull(title)

#top 10 worst movies based on b_m
train_set %>%
  count(movieId) %>% 
  left_join(movie_regn, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(b_reg_m) %>% 
  slice(1:10) %>% 
  pull(title)

#predicting b_m for test_set
b_reg_m_hat<-test_set%>%left_join(movie_regn, by='movieId')%>%.$b_reg_m

# rating prediction and rmse of movie effect
mov_regn_rating<-mu_train+b_reg_m_hat
mov_regn_rmse<-RMSE(test_set$rating, mov_regn_rating)
rmse_summary<-rbind(rmse_summary, data.frame(method = "Regularization Movie Bias model", RMSE = mov_regn_rmse))
rmse_summary %>% knitr::kable()


#Fine tune model to get optimum lambda and extend to model user efefcts
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu<- mean(train_set$rating)
  b_m <- train_set %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_m, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu)/(n()+l))
  ratings_hat <- 
    test_set %>% 
    left_join(b_m, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_m + b_u) %>%
    .$pred
  return(RMSE(ratings_hat, test_set$rating))
})

#Plot of lambdas Vs rmse
qplot(lambdas, rmses)  

#Choosing the optimum lambda value
lambda <- lambdas[which.min(rmses)]
lambda

#Choosing the minimum RMSE
mov_user_regn_rmse<-min(rmses)
rmse_summary<-rbind(rmse_summary, data.frame(method = "Regularization Movie and User Bias model", RMSE = mov_user_regn_rmse))
rmse_summary %>% knitr::kable()

## Genre effect model
# Observe genre behavior charts here
train_set %>% group_by(genres) %>% 
  summarize(g_avg = mean(rating))%>%filter(n()>=100)%>%
  ggplot(aes(g_avg))+geom_histogram(color="black", bins=30)+
  ggtitle("Histogram of ratings with genres")+xlab("Average Rating")+ylab("Number of ratings")

#Top 10 best genres
train_set %>% group_by(genres) %>% 
  summarize(g_avg = mean(rating))%>%
  arrange(desc(g_avg))%>%
  slice(1:10)
  
#Top 10 worst genres
train_set %>% group_by(genres) %>% 
  summarize(g_avg = mean(rating))%>%
  arrange(g_avg)%>%
  slice(1:10)

# fine tuning of lambda for genre effect
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu<- mean(train_set$rating)
  b_m <- train_set %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_m, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu)/(n()+l))
  b_g <- train_set %>% 
    left_join(b_m, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_m - b_u - mu)/(n()+l))
  ratings_hat <- 
    test_set %>% 
    left_join(b_m, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_m + b_u + b_g) %>%
    .$pred
  return(RMSE(ratings_hat, test_set$rating))
})

#Plot of lambdas Vs rmse
qplot(lambdas, rmses)  

#Choosing the optimum lambda value
lambda <- lambdas[which.min(rmses)]
lambda

#Choosing the minimum RMSE
mov_user_gen_regn_rmse<-min(rmses)
rmse_summary<-rbind(rmse_summary, data.frame(method = "Regularization Movie, User and Genre Bias model", RMSE = mov_user_gen_regn_rmse))
rmse_summary %>% knitr::kable()

# Final model: use parameters from edx set
l<-lambda # use the lambda obtained above to apply on the validation set for predicting output
mu<- mean(train_set$rating)
b_m <- train_set %>%
  group_by(movieId) %>%
  summarize(b_m = sum(rating - mu)/(n()+l))
b_u <- train_set %>% 
  left_join(b_m, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_m - mu)/(n()+l))
b_g <- train_set %>% 
  left_join(b_m, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_u - b_m - mu)/(n()+l))


# Final model application: use validation set
final_rating <- validation %>%
  left_join(b_m, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(pred = mu + b_m + b_u + b_g) %>%
  .$pred

rmse_validation <- RMSE(final_rating, validation$rating)

#Final RMSE to be reported
rmse_validation
