##Script create dataframe of US congressional representatives and their data for migration into db. Output saves into csv.
##Author: Eleonore Fournier-Tombs
##Last update: 27 June 2015

#1. Acccess data from GovTrack API (Representatives from Last Congress)

library(jsonlite)
library(curl)
FullList <- fromJSON("https://www.govtrack.us/api/v2/role?current=true&role_type=representative&limit=441")

##Refine data

MetaList <- FullList$objects
PersonList <- FullList$objects$person

USReps <- data.frame(matrix(0))

USReps <- cbind(USReps,MetaList[,c(17,5,18,6,9)])

USReps <-cbind(USReps, PersonList[,c(2,4,10,8,16,5,17)])

USReps <- within(USReps, matrix.0. <- paste(state, district, sep = "-")) ## Set unique ID for migration into db


##Print dataframe into csv
write.csv(USReps, "usreps.csv")