##Script to pull Twitter user information from Twitter API for all US Representatives who have a Twitter ID
##Author: Eleonore Fournier-Tombs
##Last update: 27 June 2015

library(twitteR)
source('credentials.R', local=T) ##Load Twitter OAuth credentials
source('representatives.r', local=T)  ##Load list of US Representatives

setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_secret)

## getCurRateLimit() #There is a limit of number of queries to the Twitter API of 180 per 15 minutes. Since there are
## over 400 representatives in our list, the list has to be split and run several times. 

USRepsTwitter <- USReps$twitterid
USRepsTwitter <- USRepsTwitter[!is.na(USRepsTwitter)]

USRepsTwiUsers <- data.frame(matrix(0, ncol=17, nrow = 0))

for (i in USRepsTwitter[1:179]) {
      tryCatch({user <- getUser(i) ## Skip over suspended accounts
      frame <- user$toDataFrame()
      USRepsTwiUsers[i,] <- frame
      },
      error = function(e) {} #TO DO:Print error in error log
      )
}

colnames(USRepsTwiUsers) <- colnames(frame)

## STOP SCRIPT AND RUN BELOW AFTER TIME LIMIT (15 MINUTES) HAS PASSED
for (i in USRepsTwitter[180:359]) { 
      tryCatch({ 
            user <- getUser(i)
      frame <- user$toDataFrame()
      USRepsTwiUsers <- rbind(USRepsTwiUsers, frame)
      },
      error = function(e) {}
      )
}

## STOP SCRIPT AND RUN BELOW AFTER TIME LIMIT (15 MINUTES) HAS PASSED
for (i in USRepsTwitter[355:417]) {
      tryCatch({user <- getUser(i)
      frame <- user$toDataFrame()
      USRepsTwiUsers <- rbind(USRepsTwiUsers, frame)
      },
      error = function(e) {}
      )
}


##Print dataframe into csv
write.csv(USRepsTwiUsers, "usrepstwitter.csv")

##TwitterR Notes:
#User object fields: 'name', 'screenName', 'id', 'lastStatus', 'description', 'statuses Count', 'followersCount'
# 'favoritesCount', 'friendsCount', 'url', 'created', 'protected', 'verified', 'location', 'listedCount', 'followRequestSent', 'profileImageUrl'
#Methods: 'getFollowersIDs', 'getFollowers', 'getFriendIDs', 'getFriends', 'toDataFrame'